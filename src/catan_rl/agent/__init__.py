"""
Agent module following the LlamaGym Agent pattern.

Provides:
- CatanAgent: Abstract base class defining the agent interface
- QwenCatanAgent: Concrete implementation for Qwen3-8B-Instruct
- Observation formatting, action parsing, and system prompts
"""

from .base import CatanAgent, AgentAction
from .qwen_agent import QwenCatanAgent

__all__ = ["CatanAgent", "AgentAction", "QwenCatanAgent"]
