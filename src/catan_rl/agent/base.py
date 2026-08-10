"""
CatanAgent — Abstract base class following the LlamaGym Agent pattern.

This module defines the core agent interface for Catan AI players.
The design follows LlamaGym's Agent abstract class pattern (get_system_prompt,
format_observation, extract_action) but is decoupled from any specific training
algorithm (PPO/GRPO). This separation allows the agent to be used for both
data collection (rollouts) and training with different RL algorithms.

Key design decisions:
- All game-state formatting logic is centralized in format_observation()
- Action parsing uses multiple fallback strategies for robustness
- System prompt is configurable per training phase (basic rules vs advanced strategy)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """
    Structured action output from the agent.

    Attributes:
        action_index: The integer index into the valid_actions list
        action_type: String name of the action type (e.g., "BUILD_SETTLEMENT")
        params: Dict of action-specific parameters
        raw_text: The original model output text (for debugging)
        is_valid: Whether this action passed validation against valid_actions
    """
    action_index: int
    action_type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    is_valid: bool = True


@dataclass
class TurnRecord:
    """
    A single turn record, stored in the episode trajectory.
    """
    prompt: str                          # Full prompt sent to model
    completion: str                      # Raw model output
    action: Optional[AgentAction]        # Parsed action
    reward: float = 0.0                  # Reward assigned
    game_phase: str = ""                 # Current game phase
    turn_number: int = 0                 # Turn index
    victory_points: int = 0              # Agent's VPs at this turn


class CatanAgent(ABC):
    """
    Abstract base class for Catan-playing LLM agents.

    Follows the LlamaGym Agent pattern with three abstract methods
    that subclasses must implement:
      - get_system_prompt() -> str
      - format_observation(game_state, valid_actions, player_index) -> str
      - extract_action(response, valid_actions) -> AgentAction

    The training loop (GRPO via TRL) is managed externally. This class focuses
    purely on the agent-environment interface: taking observations, generating
    actions, and recording rewards.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the agent.

        Args:
            model: A HuggingFace model (AutoModelForCausalLM) or None for testing
            tokenizer: HuggingFace tokenizer with chat template
            device: "cuda", "cpu", or "auto"
            config: Optional config dict for generation parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or {}

        # Generation parameters (can be overridden via config)
        self.max_new_tokens = self.config.get("max_new_tokens", 128)
        self.temperature = self.config.get("temperature", 0.8)
        self.top_p = self.config.get("top_p", 0.9)
        self.top_k = self.config.get("top_k", 50)
        self.do_sample = self.config.get("do_sample", True)

        # Episode state
        self.conversation_history: List[Dict[str, str]] = []
        self.current_trajectory: List[TurnRecord] = []
        self.total_reward: float = 0.0
        self.turn_count: int = 0

        # Statistics
        self.stats = {
            "total_actions": 0,
            "valid_actions": 0,
            "invalid_actions": 0,
            "parse_failures": 0,
        }

    # =========================================================================
    # Abstract methods — subclasses MUST implement these
    # =========================================================================

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt that defines the agent's role, rules, and strategy.

        The system prompt should include:
        - Catan game rules summary
        - Resource types and building costs
        - Victory conditions (settlements=1VP, cities=2VP, longest road=2VP, etc.)
        - Strategic heuristics (prioritize 6/8 tiles, resource diversity, harbor access)
        - Output format specification (JSON with action_number field)

        Returns:
            str: The complete system prompt
        """
        ...

    @abstractmethod
    def format_observation(
        self,
        game_state: Any,
        valid_actions: List[Any],
        player_index: int = 0,
    ) -> str:
        """
        Convert the raw game state and valid actions into a structured text
        observation that the LLM can understand and reason over.

        This is the MOST CRITICAL method for agent quality. The observation
        should be structured, informative, and easy for the model to parse.

        Args:
            game_state: The catanatron State object (env.game.state)
            valid_actions: List of valid catanatron Action objects
            player_index: Index of the agent's player (0, 1, 2, or 3)

        Returns:
            str: Formatted observation text
        """
        ...

    @abstractmethod
    def extract_action(
        self,
        response: str,
        valid_actions: List[Any],
    ) -> AgentAction:
        """
        Parse the model's text response into a validated AgentAction.

        Should implement multiple parsing strategies with fallbacks:
        1. Try JSON parse with "action_number" field
        2. Try JSON parse with "action" + "params" fields
        3. Try regex to find action type name
        4. Fuzzy string match against valid action descriptions
        5. Fallback: return random valid action (with warning)

        Args:
            response: Raw text output from the model
            valid_actions: List of valid catanatron Action objects

        Returns:
            AgentAction with is_valid=True if parsing succeeded
        """
        ...

    # =========================================================================
    # Concrete methods — shared across all implementations
    # =========================================================================

    def build_prompt(self, observation_text: str) -> str:
        """
        Construct the full prompt from system prompt and observation.

        For chat models (like Qwen3-8B-Instruct), uses the tokenizer's
        chat template. For base models, uses a simple concatenation.

        Args:
            observation_text: Formatted observation from format_observation()

        Returns:
            str: Complete prompt ready for tokenization
        """
        system_prompt = self.get_system_prompt()

        if self.tokenizer and hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            # Use chat template for instruction-tuned models
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": observation_text},
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, falling back to simple format")

        # Simple format fallback (for base models or older tokenizers)
        prompt = f"System: {system_prompt}\n\nUser: {observation_text}\n\nAssistant:"
        return prompt

    def build_chat_messages(self, observation_text: str) -> List[Dict[str, str]]:
        """
        Build chat message list for the current observation.

        Args:
            observation_text: Formatted observation

        Returns:
            List of message dicts [{"role": ..., "content": ...}]
        """
        system_prompt = self.get_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation_text},
        ]
        return messages

    def act(self, observation: Any, valid_actions: List[Any] = None, player_index: int = 0) -> AgentAction:
        """
        Generate an action given the current game state.

        This is the main interaction method. It:
        1. Formats the observation into text
        2. Builds the prompt (system + observation)
        3. Generates a response from the model
        4. Parses the response into an AgentAction

        Args:
            observation: Raw observation from the environment
            valid_actions: List of valid Action objects (if None, tries to get from observation)
            player_index: Index of the agent's player

        Returns:
            AgentAction with the chosen action
        """
        # If observation is a dict, extract game_state and valid_actions
        if isinstance(observation, dict):
            game_state = observation.get("game_state", observation.get("state"))
            if valid_actions is None:
                valid_actions = observation.get("valid_actions", [])
        else:
            game_state = observation

        if valid_actions is None:
            valid_actions = []

        # Format observation
        obs_text = self.format_observation(game_state, valid_actions, player_index)

        # Build prompt
        prompt = self.build_prompt(obs_text)

        # Tokenize and generate
        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            # No model loaded — return a random valid action for testing
            import random
            if valid_actions:
                action_index = random.randint(0, len(valid_actions) - 1)
                return AgentAction(
                    action_index=action_index,
                    action_type=str(valid_actions[action_index]),
                    raw_text="[NO_MODEL] random action",
                    is_valid=True,
                )
            return AgentAction(action_index=0, action_type="END_TURN", raw_text="[NO_MODEL]", is_valid=True)

        # Extract action
        agent_action = self.extract_action(response, valid_actions)

        # Update trajectory
        self.current_trajectory.append(TurnRecord(
            prompt=prompt,
            completion=response,
            action=agent_action,
        ))

        # Update stats
        self.stats["total_actions"] += 1
        if agent_action.is_valid:
            self.stats["valid_actions"] += 1
        else:
            self.stats["invalid_actions"] += 1

        self.turn_count += 1

        return agent_action

    def assign_reward(self, reward: float) -> None:
        """
        Record a reward for the most recent action in the trajectory.

        Following the LlamaGym pattern, this records the reward and
        accumulates the total episode reward.

        Args:
            reward: Scalar reward value
        """
        if self.current_trajectory:
            self.current_trajectory[-1].reward = reward
        self.total_reward += reward

    def terminate_episode(self) -> Dict[str, Any]:
        """
        End the current episode and return trajectory data.

        In LlamaGym, this triggers PPO training when enough episodes are
        collected. In our implementation, this just organizes and returns
        the trajectory data — the external training loop handles GRPO.

        Returns:
            Dict with trajectory data, stats, and total reward
        """
        trajectory_data = {
            "records": [r for r in self.current_trajectory],
            "total_reward": self.total_reward,
            "num_turns": self.turn_count,
            "stats": dict(self.stats),
        }

        # Reset episode state
        self.reset_episode()

        return trajectory_data

    def reset_episode(self) -> None:
        """Reset episode state for a new game."""
        self.conversation_history = []
        self.current_trajectory = []
        self.total_reward = 0.0
        self.turn_count = 0

    def get_validity_rate(self) -> float:
        """
        Get the rate of valid actions produced by the agent.

        Returns:
            float: valid_actions / total_actions (0.0 if no actions)
        """
        if self.stats["total_actions"] == 0:
            return 0.0
        return self.stats["valid_actions"] / self.stats["total_actions"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"validity_rate={self.get_validity_rate():.2%})"
        )


# Import torch at module level for type hints
try:
    import torch
except ImportError:
    torch = None
