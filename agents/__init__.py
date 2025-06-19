"""
Agents package for compensation planning system.
Contains the offer chain and other agent-related functionality.
"""

from .offer_chain import run_compensation_planner, generate_completion

__all__ = ['run_compensation_planner', 'generate_completion']