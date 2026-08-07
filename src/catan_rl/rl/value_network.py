# catanatron_experimental/catanatron_experimental/rl_value_network.py

import os
import numpy as np
import torch
import torch.nn as nn


class CatanValueNetwork(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self._hidden_dims = list(hidden_dims)  # stored explicitly for save/load

        layers = []
        prev_dim = input_dim
        for hidden_dim in self._hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def predict(self, game, color) -> float:
        features = extract_features(game, color)
        x = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            return self.forward(x).item()

    def save(self, path: str):
        torch.save({
            "state_dict":  self.state_dict(),
            "input_dim":   self.network[0].in_features,
            "hidden_dims": self._hidden_dims,  # saved explicitly
        }, path)
        print(f"  [RL] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CatanValueNetwork":
        checkpoint  = torch.load(path, map_location="cpu")
        input_dim   = checkpoint["input_dim"]
        hidden_dims = checkpoint.get("hidden_dims", None)

        # If hidden_dims not in checkpoint (old save format), infer from weights
        if hidden_dims is None:
            hidden_dims = []
            for name, param in checkpoint["state_dict"].items():
                # Only pick up Linear weight layers, not LayerNorm
                if name.endswith(".weight"):
                    # Linear layers have 2D weight, LayerNorm has 1D
                    if len(param.shape) == 2:
                        hidden_dims.append(param.shape[0])
            hidden_dims = hidden_dims[:-1]  # exclude output layer
            print(f"  [RL] Inferred hidden_dims from weights: {hidden_dims}")

        model = cls(input_dim, hidden_dims)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"  [RL] Model loaded from {path} "
              f"(input_dim={input_dim}, hidden_dims={hidden_dims})")
        return model


# ------------------------------------------------------------------ #
#  Feature extraction                                                  #
# ------------------------------------------------------------------ #

def extract_features(game, color) -> np.ndarray:
    state       = game.state
    colors      = state.colors
    idx         = colors.index(color)
    num_players = len(colors)
    features    = []

    # VP
    my_vp = state.player_state.get(f"P{idx}_VICTORY_POINTS", 0)
    features.append(my_vp / 10.0)

    # VP gap vs leader
    other_vps = [
        state.player_state.get(f"P{i}_VICTORY_POINTS", 0)
        for i in range(num_players) if i != idx
    ]
    leader_vp = max(other_vps) if other_vps else 0
    features.append((my_vp - leader_vp) / 10.0)

    # Resources
    for r in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]:
        amt = state.player_state.get(f"P{idx}_{r}_IN_HAND", 0)
        features.append(min(amt / 10.0, 1.0))

    # Total hand size
    total_hand = sum(
        state.player_state.get(f"P{idx}_{r}_IN_HAND", 0)
        for r in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    )
    features.append(min(total_hand / 15.0, 1.0))

    # Buildings
    features.append((5  - state.player_state.get(f"P{idx}_SETTLEMENTS_AVAILABLE", 5))  / 5.0)
    features.append((4  - state.player_state.get(f"P{idx}_CITIES_AVAILABLE",      4))  / 4.0)
    features.append((15 - state.player_state.get(f"P{idx}_ROADS_AVAILABLE",       15)) / 15.0)

    # Dev cards
    for card in ["KNIGHT", "YEAR_OF_PLENTY", "ROAD_BUILDING", "MONOPOLY", "VICTORY_POINT"]:
        amt = state.player_state.get(f"P{idx}_{card}_IN_HAND", 0)
        features.append(min(amt / 5.0, 1.0))

    # Knights played
    features.append(min(state.player_state.get(f"P{idx}_PLAYED_KNIGHT", 0) / 10.0, 1.0))

    # Special cards
    try:
        from catanatron.state_functions import get_longest_road_color, get_largest_army_color
        longest_color = get_longest_road_color(state)
        largest_color = get_largest_army_color(state)
    except Exception:
        longest_color = getattr(state, "longest_road_color", None)
        largest_color = getattr(state, "largest_army_color", None)

    features.append(1.0 if longest_color == color else 0.0)
    features.append(1.0 if largest_color == color else 0.0)

    # Turn number
    features.append(min(state.num_turns / 200.0, 1.0))

    # Opponent VPs (up to 3, padded)
    opp_vps = sorted(other_vps, reverse=True)
    for i in range(3):
        features.append(opp_vps[i] / 10.0 if i < len(opp_vps) else 0.0)

    # Production score
    try:
        features.append(min(_compute_production_score(game, color) / 20.0, 1.0))
    except Exception:
        features.append(0.0)

    # Port access
    try:
        features.append(_compute_port_score(game, color) / 5.0)
    except Exception:
        features.append(0.0)

    # --- Reachable settlement spots via existing roads ---
    try:
        reachable = _count_reachable_settlement_spots(game, color)
        features.append(min(reachable / 10.0, 1.0))
    except Exception:
        features.append(0.0)

    # --- Current road count ---
    try:
        road_len = _count_roads(game, color)
        features.append(min(road_len / 15.0, 1.0))
    except Exception:
        features.append(0.0)

    # --- VP diff vs each opponent (up to 3, padded) ---
    opp_indices = [i for i in range(num_players) if i != idx]
    for i in opp_indices[:3]:
        opp_vp = state.player_state.get(f"P{i}_VICTORY_POINTS", 0)
        features.append((my_vp - opp_vp) / 10.0)
    for _ in range(3 - len(opp_indices[:3])):
        features.append(0.0)

    return np.array(features, dtype=np.float32)

def _compute_production_score(game, color) -> float:
    PIP_VALUES = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    score = 0.0
    board = game.state.board
    for node_id, (bcolor, btype) in board.buildings.items():
        if bcolor != color:
            continue
        multiplier = 2 if btype == "CITY" else 1
        try:
            for tile in board.map.adjacent_tiles[node_id]:
                if hasattr(tile, "number") and tile.number:
                    score += PIP_VALUES.get(tile.number, 0) * multiplier
        except Exception:
            pass
    return score


def _compute_port_score(game, color) -> float:
    score = 0.0
    board = game.state.board
    try:
        for node_id, (bcolor, _) in board.buildings.items():
            if bcolor != color:
                continue
            if node_id in board.map.port_nodes:
                score += 1.0
    except Exception:
        pass
    return score


def get_feature_dim() -> int:
    # 1 vp + 1 vp_gap + 5 resources + 1 hand +
    # 3 buildings + 5 dev_cards + 1 knights +
    # 2 special + 1 turn + 3 opp_vps + 1 production + 1 port = 25
    return 30

def _count_reachable_settlement_spots(game, color) -> int:
    """Count empty nodes reachable by extending current roads."""
    board = game.state.board
    reachable = set()
    for (n1, n2), road_color in board.roads.items():
        if road_color != color:
            continue
        for node in [n1, n2]:
            if node in board.buildings:
                continue
            neighbors = board.map.adjacent_nodes.get(node, [])
            if not any(n in board.buildings for n in neighbors):
                reachable.add(node)
    return len(reachable)


def _count_roads(game, color) -> int:
    """Count total roads placed by color."""
    return sum(1 for c in game.state.board.roads.values() if c == color)