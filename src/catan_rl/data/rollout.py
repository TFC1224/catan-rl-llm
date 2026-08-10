"""
Game rollout for data collection.

Provides functions for playing full games with an agent and recording
trajectories for GRPO training.

Key insight: CatanatronEnv.step() internally calls _advance_until_p0_decision(),
which auto-plays all non-P0 (bot) turns. The game loop simply needs to:
  1. Get P0's game state (always P0's turn after step/reset)
  2. Agent decides an action
  3. Map sequential index → action space index
  4. env.step(action_space_index) — executes P0 action + auto-advances opponents
  5. Repeat until done
"""

import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def play_game_with_agent(
    agent: Any,
    env: Any,
    player_index: int = 0,
    record_trajectory: bool = True,
) -> Dict[str, Any]:
    """
    Play one full game with the given agent.

    The agent always plays as P0 (BLUE). CatanatronEnv auto-advances
    opponent turns via _advance_until_p0_decision().

    Args:
        agent: A CatanAgent instance
        env: A CatanatronEnv instance (must be reset before calling)
        player_index: Agent's player index (0 = BLUE, P0)
        record_trajectory: If True, record (state, action, outcome) records

    Returns:
        Dict with:
        - "outcome": "WIN", "LOSS", or "DRAW"
        - "records": list of game state records (if record_trajectory=True)
        - "total_reward": float
        - "num_turns": int
        - "agent_vp": int (final VP)
    """
    agent.reset_episode()

    records = []
    obs = env.reset()
    done = False
    turn = 0
    agent_vp = 0

    while not done and turn < 200:
        state = env.game.state
        int_actions = env.get_valid_actions()
        playable_actions = list(state.playable_actions)

        if not int_actions or not playable_actions:
            break

        # Agent decides (sequential index into playable_actions)
        agent_action = agent.act(
            observation=state,
            valid_actions=playable_actions,
            player_index=player_index,
        )

        # Map sequential index to action space index for env.step()
        seq_idx = agent_action.action_index
        if seq_idx is None or seq_idx < 0 or seq_idx >= len(int_actions):
            seq_idx = 0
        action_idx = int_actions[seq_idx]

        # Record game state before stepping
        if record_trajectory:
            record = _create_rollout_record(
                env=env,
                game_state=state,
                playable_actions=playable_actions,
                int_actions=int_actions,
                agent_action=agent_action,
                turn_number=turn,
                player_index=player_index,
            )
            if record:
                records.append(record)

        # Step environment — executes P0 action + auto-advances opponents
        obs, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated

        # Record reward
        agent.assign_reward(reward)

        # Track agent's VP
        try:
            agent_vp = env.game.state.player_state.get(
                f"P{player_index}_ACTUAL_VICTORY_POINTS", agent_vp
            )
        except Exception:
            pass

        turn += 1

    # Determine outcome
    outcome = _determine_outcome(env, player_index)

    # Close trajectory
    trajectory = agent.terminate_episode()

    return {
        "outcome": outcome,
        "records": records,
        "total_reward": trajectory.get("total_reward", 0.0),
        "num_turns": turn,
        "agent_vp": agent_vp,
    }


def _create_rollout_record(
    env: Any,
    game_state: Any,
    playable_actions: List[Any],
    int_actions: List[int],
    agent_action: Any,
    turn_number: int,
    player_index: int,
) -> Optional[Dict]:
    """Create a rollout record from game state and agent action."""
    try:
        from ..agent.observation import format_catan_observation
        from ..agent.prompts import get_system_prompt

        obs_text = format_catan_observation(
            game_state=game_state,
            valid_actions=playable_actions,
            player_index=player_index,
            verbose=True,
        )

        # Build the full prompt with chat template (system + user + assistant prefix)
        system_prompt = get_system_prompt(
            version="v1",
            player_color="BLUE",
            vps_to_win=env.game.vps_to_win if hasattr(env.game, 'vps_to_win') else 6,
        )

        # Build the FULL prompt with chat template applied.
        # GRPOTrainer will use this as-is (apply_chat_template=False).
        # Format: <|im_start|>system\n{...}\n<|im_end|>\n<|im_start|>user\n{...}\n<|im_end|>\n<|im_start|>assistant\n
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": obs_text},
        ]
        # Use the tokenizer from the agent if available
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                '/root/autodl-tmp/Qwen/Qwen3-8B/',
                trust_remote_code=True,
            )
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: simple concatenation
            prompt_text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{obs_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # Serialize the FULL game (not just state) for simulation
        game_bytes = pickle.dumps(env.game)

        # Format valid actions and action space indices for the reward function
        import json
        valid_actions_json = json.dumps([str(a) for a in playable_actions])
        int_actions_json = json.dumps(int_actions)

        return {
            "prompt": prompt_text,
            "serialized_game": game_bytes,
            "valid_actions": valid_actions_json,
            "int_actions": int_actions_json,
            "chosen_action_index": agent_action.action_index,
            "chosen_action_text": agent_action.raw_text,
            "turn_number": turn_number,
            "phase": str(game_state.current_prompt),
        }
    except Exception as e:
        logger.warning(f"Failed to create rollout record: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to create rollout record: {e}")
        return None


def _determine_outcome(env: Any, player_index: int) -> str:
    """Determine game outcome for the agent."""
    try:
        winner_color = env.game.winning_color()
        if winner_color is None:
            return "DRAW"

        winner_str = str(winner_color)
        agent_color_map = {0: "BLUE", 1: "RED", 2: "WHITE", 3: "ORANGE"}
        agent_color = agent_color_map.get(player_index, "BLUE")

        if agent_color in winner_str.upper():
            return "WIN"
        else:
            return "LOSS"
    except Exception:
        return "DRAW"
