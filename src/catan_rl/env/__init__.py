"""
Environment module for Catanatron Gym integration.

Provides:
- make_catan_env: Factory function for creating configured Catan environments
- Game state serialization/deserialization
- Fast parallel game simulation for reward computation
- Reward function implementations
"""

from .catan_env import make_catan_env

__all__ = ["make_catan_env"]
