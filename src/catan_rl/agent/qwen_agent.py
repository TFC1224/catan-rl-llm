"""
QwenCatanAgent — Concrete implementation using Qwen3-8B-Instruct.

This class implements the CatanAgent abstract interface for Qwen3-8B.
It handles model loading (4-bit QLoRA), LoRA adapter loading, tokenizer
setup, generation, and the three LlamaGym abstract methods.

Key features:
- 4-bit quantization via BitsAndBytes for memory efficiency
- Flash Attention 2 support for faster inference
- Proper chat template usage for Qwen3-8B-Instruct
- LoRA adapter loading for checkpointed models
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from .base import CatanAgent, AgentAction
from .prompts import get_system_prompt
from .observation import format_catan_observation
from .action_parser import parse_action

logger = logging.getLogger(__name__)


class QwenCatanAgent(CatanAgent):
    """
    Catan agent powered by Qwen3-8B-Instruct.

    Uses 4-bit QLoRA for memory efficiency. Supports loading LoRA adapters
    from checkpoints for continued training or evaluation.

    Usage:
        agent = QwenCatanAgent.from_pretrained(
            model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
            device="cuda",
        )
        action = agent.act(observation, valid_actions)
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None,
        lora_path: Optional[str] = None,
        prompt_version: str = "v1",
    ):
        """
        Initialize the Qwen Catan agent.

        Args:
            model: Pre-loaded model or None
            tokenizer: Pre-loaded tokenizer or None
            device: Device for inference
            config: Configuration dict
            lora_path: Optional path to LoRA adapter checkpoint
            prompt_version: System prompt version ("v1", "v2", "concise")
        """
        super().__init__(model, tokenizer, device, config)
        self.prompt_version = prompt_version
        self.lora_path = lora_path

        # Lazy-loaded attributes
        self._player_index: int = 0
        self._vps_to_win: int = 6

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen3-8B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        lora_path: Optional[str] = None,
        prompt_version: str = "v1",
        **kwargs,
    ) -> "QwenCatanAgent":
        """
        Factory method: load model and tokenizer from HuggingFace.

        Args:
            model_name: HuggingFace model name
            device: "cuda", "cpu", or "auto"
            load_in_4bit: Use 4-bit quantization
            lora_path: Optional path to LoRA adapter
            prompt_version: System prompt version

        Returns:
            Initialized QwenCatanAgent
        """
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        logger.info(f"Loading model: {model_name}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model loading config
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else device,
        }

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            logger.info("  Using 4-bit QLoRA quantization")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        # Load LoRA adapter if provided
        if lora_path:
            from peft import PeftModel
            logger.info(f"  Loading LoRA adapter: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            logger.info("  LoRA adapter loaded")

        agent = cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_version=prompt_version,
            lora_path=lora_path,
            **kwargs,
        )

        return agent

    def configure_episode(self, player_index: int = 0, vps_to_win: int = 6) -> None:
        """
        Configure agent for a new episode.

        Args:
            player_index: Agent's player index in the game (0-3)
            vps_to_win: Victory points needed to win
        """
        self._player_index = player_index
        self._vps_to_win = vps_to_win

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def get_system_prompt(self) -> str:
        """
        Return the system prompt for Qwen3-8B-Instruct.

        Uses the configured prompt version and includes player info.
        """
        # Determine player color name
        color_map = {0: "BLUE", 1: "RED", 2: "WHITE", 3: "ORANGE"}
        player_color = color_map.get(self._player_index, "BLUE")

        return get_system_prompt(
            version=self.prompt_version,
            player_color=player_color,
            vps_to_win=self._vps_to_win,
        )

    def format_observation(
        self,
        game_state: Any,
        valid_actions: List[Any],
        player_index: int = 0,
    ) -> str:
        """
        Format the Catanatron game state for Qwen3-8B-Instruct.

        Uses the structured observation formatter with all sections:
        phase, resources, dev cards, buildings, VPs, board, actions.
        """
        return format_catan_observation(
            game_state=game_state,
            valid_actions=valid_actions,
            player_index=player_index,
            verbose=True,
        )

    def extract_action(
        self,
        response: str,
        valid_actions: List[Any],
    ) -> AgentAction:
        """
        Parse Qwen3-8B-Instruct output into a validated AgentAction.

        Handles typical Qwen chat output formats (JSON blocks, code fences).
        """
        # Clean response — remove common Qwen artifacts
        cleaned = self._clean_response(response)

        # Parse using the multi-strategy parser
        agent_action = parse_action(cleaned, valid_actions)

        return agent_action

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _clean_response(self, response: str) -> str:
        """
        Clean the raw model output.

        Qwen3-8B-Instruct sometimes outputs:
        - JSON in ```json fences
        - Extra text before/after the JSON
        - Markdown formatting
        - Thinking tokens (if thinking is enabled)

        Args:
            response: Raw model output

        Returns:
            Cleaned response string
        """
        cleaned = response.strip()

        # Remove markdown code fences
        if "```json" in cleaned:
            # Extract content between ```json and ```
            match = __import__("re").search(
                r"```json\s*([^`]+)```", cleaned
            )
            if match:
                cleaned = match.group(1).strip()
        elif "```" in cleaned:
            match = __import__("re").search(r"```\s*([^`]+)```", cleaned)
            if match:
                cleaned = match.group(1).strip()

        # Remove <｜end▁of▁thinking｜> tags
        cleaned = __import__("re").sub(r"<[^>]+>", "", cleaned)

        # Remove common prefixes
        prefixes = ["Assistant:", "assistant:", "Response:", "response:"]
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        return cleaned.strip()

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model given a prompt.

        Args:
            prompt: The complete prompt string (including chat template)

        Returns:
            Generated text response
        """
        if self.model is None or self.tokenizer is None:
            return '{"action_number": 0}'

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("max_prompt_length", 2048),
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response

    def __repr__(self) -> str:
        return (
            f"QwenCatanAgent("
            f"device={self.device}, "
            f"prompt_version={self.prompt_version}, "
            f"lora_path={'yes' if self.lora_path else 'no'}, "
            f"validity_rate={self.get_validity_rate():.2%})"
        )
