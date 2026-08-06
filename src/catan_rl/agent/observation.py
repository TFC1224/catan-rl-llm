"""
Game state observation formatter for Catanatron.

This is the MOST CRITICAL module for agent quality. It translates the raw
Catanatron game state into structured natural language the LLM can understand.

Design principles (from "Agents of Change" paper):
1. Structured sections with clear headers
2. Enumerated valid actions (numbered list) for easy model parsing
3. Only show relevant state — filter noise
4. Resource counts in a consistent order
5. Board information focused on actionable data

The formatter produces output like:

    ## Game Phase
    BUILD_INITIAL_SETTLEMENT

    ## Your Resources (BLUE)
    Wood: 0 | Brick: 0 | Sheep: 0 | Wheat: 0 | Ore: 0

    ## Available Actions
    1. BUILD_SETTLEMENT at node 5
    2. BUILD_SETTLEMENT at node 7
    ...

    Reply with: {"action_number": <N>}
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def format_catan_observation(
    game_state: Any,
    valid_actions: List[Any],
    player_index: int = 0,
    verbose: bool = True,
    action_descriptions: Optional[List[str]] = None,
) -> str:
    """
    Convert Catanatron game state and valid actions into structured text.

    Args:
        game_state: The catanatron State object (env.game.state)
        valid_actions: List of valid Action objects (from state.playable_actions)
                       or integer action indices (from env.get_valid_actions())
        player_index: Index of the agent's player (0, 1, 2, or 3)
        verbose: If True, include full board info. If False, only show summary.
        action_descriptions: Optional pre-formatted action descriptions

    Returns:
        str: Formatted observation text ready for the LLM
    """
    sections = []

    # ---- 1. Game Phase ----
    phase = _format_phase(game_state)
    sections.append(f"## Game Phase\n{phase}")

    # ---- 2. Player Resources ----
    resources = _format_resources(game_state, player_index)
    sections.append(f"## Your Resources (Player {player_index})\n{resources}")

    # ---- 3. Development Cards ----
    dev_cards = _format_dev_cards(game_state, player_index)
    sections.append(f"## Your Development Cards\n{dev_cards}")

    # ---- 4. Buildings Owned ----
    buildings = _format_buildings(game_state, player_index)
    sections.append(f"## Your Buildings\n{buildings}")

    # ---- 5. Victory Points (all players) ----
    vps = _format_victory_points(game_state)
    sections.append(f"## Victory Points\n{vps}")

    # ---- 6. Board Summary (if verbose) ----
    if verbose:
        board = _format_board_summary(game_state)
        sections.append(f"## Board Summary\n{board}")

    # ---- 7. Available Actions (most important!) ----
    actions = _format_actions(valid_actions)
    sections.append(f"## Available Actions (choose one number)\n{actions}")

    # ---- 8. Output Format Reminder ----
    sections.append('Reply with ONLY: {{"action_number": <integer>}}')

    return "\n\n".join(sections)


def _format_phase(game_state: Any) -> str:
    """Format the current game phase/prompt."""
    try:
        prompt = str(game_state.current_prompt)
        # Clean up enum names
        if hasattr(game_state.current_prompt, 'name'):
            prompt = game_state.current_prompt.name
        # Make readable
        prompt = prompt.replace('_', ' ').title()
        return prompt
    except Exception as e:
        logger.warning(f"Failed to format phase: {e}")
        return "PLAY_TURN"


def _format_resources(game_state: Any, player_index: int) -> str:
    """Format the player's resource counts."""
    try:
        ps = game_state.player_state
        pfx = f"P{player_index}"

        resources = [
            ("Wood", ps.get(f"{pfx}_WOOD_IN_HAND", 0)),
            ("Brick", ps.get(f"{pfx}_BRICK_IN_HAND", 0)),
            ("Sheep", ps.get(f"{pfx}_SHEEP_IN_HAND", 0)),
            ("Wheat", ps.get(f"{pfx}_WHEAT_IN_HAND", 0)),
            ("Ore", ps.get(f"{pfx}_ORE_IN_HAND", 0)),
        ]

        parts = [f"{name}: {count}" for name, count in resources]
        return " | ".join(parts)
    except Exception as e:
        logger.warning(f"Failed to format resources: {e}")
        return "Error reading resources"


def _format_dev_cards(game_state: Any, player_index: int) -> str:
    """Format the player's development cards."""
    try:
        ps = game_state.player_state
        pfx = f"P{player_index}"

        cards = [
            ("Knights", ps.get(f"{pfx}_KNIGHT_IN_HAND", 0)),
            ("Victory Points", ps.get(f"{pfx}_VICTORY_POINT_IN_HAND", 0)),
            ("Monopoly", ps.get(f"{pfx}_MONOPOLY_IN_HAND", 0)),
            ("Year of Plenty", ps.get(f"{pfx}_YEAR_OF_PLENTY_IN_HAND", 0)),
            ("Road Building", ps.get(f"{pfx}_ROAD_BUILDING_IN_HAND", 0)),
        ]

        # Count played knights for Largest Army tracking
        played_knights = ps.get(f"{pfx}_PLAYED_KNIGHT", 0)

        parts = [f"{name}: {count}" for name, count in cards]
        parts.append(f"Played Knights: {played_knights}")
        return " | ".join(parts)
    except Exception as e:
        logger.warning(f"Failed to format dev cards: {e}")
        return "Error reading development cards"


def _format_buildings(game_state: Any, player_index: int) -> str:
    """Format the player's built structures."""
    try:
        ps = game_state.player_state
        pfx = f"P{player_index}"

        roads_available = ps.get(f"{pfx}_ROADS_AVAILABLE", 15)
        settlements_available = ps.get(f"{pfx}_SETTLEMENTS_AVAILABLE", 5)
        cities_available = ps.get(f"{pfx}_CITIES_AVAILABLE", 4)

        roads_built = 15 - roads_available
        settlements_built = 5 - settlements_available
        cities_built = 4 - cities_available

        # Longest road check
        longest_road_len = ps.get(f"{pfx}_LONGEST_ROAD_LENGTH", 0)
        has_road = ps.get(f"{pfx}_HAS_ROAD", False)
        has_army = ps.get(f"{pfx}_HAS_ARMY", False)

        parts = [
            f"Roads: {roads_built} built ({roads_available} available)",
            f"Settlements: {settlements_built} built ({settlements_available} available)",
            f"Cities: {cities_built} built ({cities_available} available)",
            f"Road Length: {longest_road_len}",
            f"Longest Road: {'Yes' if has_road else 'No'}",
            f"Largest Army: {'Yes' if has_army else 'No'}",
        ]
        return "\n".join(parts)
    except Exception as e:
        logger.warning(f"Failed to format buildings: {e}")
        return "Error reading buildings"


def _format_victory_points(game_state: Any) -> str:
    """Format victory points for all players."""
    try:
        ps = game_state.player_state
        lines = []

        for i in range(4):  # Up to 4 players
            vp = ps.get(f"P{i}_ACTUAL_VICTORY_POINTS", None)
            if vp is None:
                break
            # Determine color name
            color_map = {0: "BLUE", 1: "RED", 2: "WHITE", 3: "ORANGE"}
            color = color_map.get(i, f"Player{i}")
            lines.append(f"{color}: {vp} VP")

        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to format VPs: {e}")
        return "Error reading victory points"


def _format_board_summary(game_state: Any) -> str:
    """Format a summary of the board state."""
    try:
        board = game_state.board
        lines = []

        # Robber position
        robber = getattr(board, 'robber_coordinate', None)
        if robber is not None:
            lines.append(f"Robber at: {robber}")

        # Port access (if inferable)
        # This is a simplified summary; full board analysis is expensive

        return "\n".join(lines) if lines else "No board data available"
    except Exception as e:
        logger.warning(f"Failed to format board: {e}")
        return "Error reading board"


def _format_actions(valid_actions: List[Any]) -> str:
    """
    Format the list of valid actions as a numbered list.

    This is the most critical part of the observation — the model must
    choose exactly one action number from this list.

    Args:
        valid_actions: List of catanatron Action objects

    Returns:
        Numbered list string
    """
    try:
        lines = []
        for i, action in enumerate(valid_actions):
            # Get a human-readable description of the action
            desc = _describe_action(action)
            lines.append(f"{i}: {desc}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to format actions: {e}")
        return "Error reading actions"


def _describe_action(action: Any) -> str:
    """
    Convert a catanatron Action into a human-readable description.

    Handles two formats:
    1. Action namedtuple: Action(color, action_type, value)
       -> "Color ACTION_TYPE value"
    2. Integer: action space index
       -> "Action <N>" (fallback)

    Args:
        action: A catanatron Action namedtuple or int

    Returns:
        str: Human-readable action description
    """
    try:
        # Handle Action namedtuple (from state.playable_actions)
        if hasattr(action, '_fields') and hasattr(action, 'action_type'):
            color_raw = str(action.color) if hasattr(action, 'color') else ''
            # Clean color: "Color.BLUE" -> "BLUE", "<Color.BLUE: 0>" -> "BLUE"
            if '.' in color_raw:
                color_raw = color_raw.rsplit('.', 1)[-1]
            if ':' in color_raw:
                color_raw = color_raw.split(':', 1)[0].strip()

            atype_raw = str(action.action_type.name) if hasattr(action.action_type, 'name') else str(action.action_type)
            value = getattr(action, 'value', '')
            return f"{color_raw} {atype_raw} (node={value})"
        elif hasattr(action, '_fields'):
            # Some other namedtuple-like object
            parts = []
            for f in action._fields:
                v = getattr(action, f)
                v_str = str(v)
                if hasattr(v, 'name'):
                    v_str = v.name
                elif '.' in str(v):
                    v_str = str(v).rsplit('.', 1)[-1]
                parts.append(f"{f}={v_str}")
            return " | ".join(parts)
        elif isinstance(action, int):
            # Integer action index — show as-is
            return f"Action index {action}"
        else:
            return str(action)
    except Exception:
        return str(action)
