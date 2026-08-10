#!/usr/bin/env python3
"""
Train a Catan Value Network using game outcomes with dense VP reward.

Key ideas from DarekYu/Catanatron + Catan Tactics:
1. Small MLP (30 features -> [256,128,64] -> win prob) — fast single-forward-pass scoring
2. Credit assignment: temporal ramp + VP bonus (+0.12 per VP gained)
3. Self-play with past model snapshots for opponent diversity
4. Trained value net replaces slow game simulation for action scoring

This gives us a fast action scorer (<1ms per action) that can discriminate
good vs bad moves, far better than our previous simulation-based approach.
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catanatron import Game, Color
from catanatron.models.player import Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.catan_rl.rl.value_network import (
    CatanValueNetwork,
    extract_features,
    get_feature_dim,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Collection Player — records (post_action_features, vp_at_decision)
# =============================================================================

class CollectionPlayer(Player):
    """
    Plays using a value network (or random at start) and collects training data.

    Decision: score each action via value net -> pick best (epsilon-greedy)
    Records: post-action features and VP for later labeling.
    """

    def __init__(self, color, model: CatanValueNetwork, epsilon: float = 0.15, device: str = "cpu"):
        super().__init__(color)
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.episode_states: list = []  # post-action features
        self.episode_vps: list = []     # VP at each decision (before action)

    def decide(self, game, playable_actions):
        actions = list(playable_actions)

        if len(actions) == 1:
            chosen = actions[0]
        elif self.model is None or random.random() < self.epsilon:
            chosen = random.choice(actions)
        else:
            best_action = actions[0]
            best_score = float("-inf")
            for action in actions:
                try:
                    gc = game.copy()
                    gc.execute(action)
                    f = extract_features(gc, self.color)
                    x = torch.FloatTensor(f).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        score = self.model(x).item()
                    if score > best_score:
                        best_score = score
                        best_action = action
                except Exception:
                    continue
            chosen = best_action

        # Record post-action state + current VP
        try:
            idx = game.state.colors.index(self.color)
            vp = game.state.player_state.get(f"P{idx}_VICTORY_POINTS", 0)
            gc = game.copy()
            gc.execute(chosen)
            features = extract_features(gc, self.color)
            self.episode_states.append(features)
            self.episode_vps.append(vp)
        except Exception:
            pass

        return chosen

    def clear_episode(self):
        self.episode_states.clear()
        self.episode_vps.clear()


# =============================================================================
# Outcome-based Value Network Trainer
# =============================================================================

class OutcomeValueTrainer:
    """
    Trains CatanValueNetwork from self-play game outcomes.

    Label formula (from DarekYu):
      win:  label = 0.3 + 0.7 * (t/n)   # ramps from 0.3 to 1.0
      loss: label = 0.7 - 0.7 * (t/n)   # ramps from 0.7 to 0.0
      + VP bonus: +0.12 per VP gained at this decision

    Curriculum:
      Phase 1 (0-30%): Random opponents — learn basic mechanics
      Phase 2 (30-70%): WeightedRandom opponents — face reasonable play
      Phase 3 (70-100%): Self-play — refine against past snapshots
    """

    def __init__(
        self,
        model_path: str = "checkpoints/rl_value/value_network.pt",
        num_episodes: int = 500,
        batch_size: int = 256,
        learning_rate: float = 5e-4,
        epsilon_start: float = 0.25,
        epsilon_end: float = 0.05,
        update_every: int = 5,
        snapshot_every: int = 100,
        pool_size: int = 5,
        buffer_capacity: int = 100000,
        load_existing: bool = False,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.update_every = update_every
        self.snapshot_every = snapshot_every
        self.pool_size = pool_size
        self.device = device

        input_dim = get_feature_dim()
        logger.info(f"Feature dimension: {input_dim}")

        self.model = CatanValueNetwork(input_dim=input_dim).to(device)

        if load_existing and os.path.exists(model_path):
            self.model = CatanValueNetwork.load(model_path).to(device)
            logger.info(f"Loaded existing model from {model_path}")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4,
        )
        self.criterion = nn.BCELoss()
        self.buffer = deque(maxlen=buffer_capacity)
        self.model_pool = []  # past snapshots for self-play
        self.losses = []
        self._current_episode = 0

    def _epsilon(self, episode: int) -> float:
        progress = episode / max(self.num_episodes, 1)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _get_opponent(self, color, episode: int) -> Player:
        """Curriculum opponent selection."""
        total = self.num_episodes
        progress = episode / max(total, 1)

        if progress < 0.3:
            # Phase 1: Random — learn basic mechanics
            return RandomPlayer(color)
        elif progress < 0.7:
            # Phase 2: WeightedRandom — face reasonable play
            return WeightedRandomPlayer(color)
        else:
            # Phase 3: Self-play against past snapshots
            if self.model_pool:
                past = random.choice(self.model_pool)
                return CollectionPlayer(color, past, epsilon=0.05, device=self.device)
            else:
                return WeightedRandomPlayer(color)

    def _collect_episode(self, colors, epsilon: float) -> Optional[dict]:
        """Run one game collecting training data."""
        shuffled = list(colors)
        random.shuffle(shuffled)

        players = []
        collection_players = {}

        for i, color in enumerate(shuffled):
            if i == 0:
                # Current model plays first color
                p = CollectionPlayer(color, self.model, epsilon, self.device)
            else:
                # Other slots get curriculum opponents
                p = self._get_opponent(color, self._current_episode)
            players.append(p)
            collection_players[color] = p

        try:
            game = Game(players)
            winner = game.play()
        except Exception as e:
            logger.warning(f"Game error: {e}")
            return None

        episode_data = {}
        for color, player in collection_players.items():
            if not isinstance(player, CollectionPlayer):
                continue
            episode_data[color] = {
                "states": player.episode_states,
                "vps": player.episode_vps,
                "outcome": 1.0 if color == winner else 0.0,
            }
        return episode_data

    def _push_episode(self, episode_data: dict):
        """Label states with temporal ramp + VP bonus."""
        if episode_data is None:
            return

        for color, data in episode_data.items():
            states = data["states"]
            vps = data.get("vps", [])
            outcome = data["outcome"]
            n = len(states)

            for t, features in enumerate(states):
                progress = t / max(n - 1, 1)

                # Temporal outcome ramp
                if outcome == 1.0:
                    base_label = 0.3 + 0.7 * progress  # 0.3 -> 1.0
                else:
                    base_label = 0.7 - 0.7 * progress  # 0.7 -> 0.0

                # VP bonus — dense reward for building progress
                vp_bonus = 0.0
                if t > 0 and t < len(vps) and vps[t] > vps[t - 1]:
                    vp_gain = vps[t] - vps[t - 1]
                    vp_bonus = 0.12 * vp_gain

                label = min(base_label + vp_bonus, 1.0)
                self.buffer.append((features, float(label)))

    def _train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        features, labels = zip(*batch)
        features_t = torch.FloatTensor(np.array(features)).to(self.device)
        labels_t = torch.FloatTensor(np.array(labels)).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(features_t)
        loss = self.criterion(preds, labels_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _health_check(self) -> dict:
        if len(self.buffer) < 64:
            return {"status": "buffer too small"}

        batch = random.sample(self.buffer, 64)
        features, labels = zip(*batch)
        features_t = torch.FloatTensor(np.array(features)).to(self.device)
        labels_np = np.array(labels)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(features_t).cpu().numpy()
        self.model.train()

        pred_std = float(np.std(preds))
        pred_mean = float(np.mean(preds))
        label_std = float(np.std(labels_np))

        if pred_std > 0.05:
            status = "STRONG"
        elif pred_std > 0.02:
            status = "moderate"
        elif pred_std > 0.01:
            status = "weak"
        else:
            status = "FLAT"

        return {
            "pred_mean": pred_mean, "pred_std": pred_std,
            "label_std": label_std, "status": status,
        }

    def _real_game_check(self, episode: int, colors):
        """Play a game using the model greedily and report decision quality."""

        class GreedyRLPlayer(Player):
            def __init__(self, color, model, device):
                super().__init__(color)
                self.model = model
                self.device = device
                self.decisions = []

            def decide(self, game, playable_actions):
                actions = list(playable_actions)
                if len(actions) == 1:
                    return actions[0]
                best_action = actions[0]
                best_score = float("-inf")
                worst_score = float("inf")
                best_idx = 0
                for i, action in enumerate(actions):
                    try:
                        gc = game.copy()
                        gc.execute(action)
                        f = extract_features(gc, self.color)
                        x = torch.FloatTensor(f).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            score = self.model(x).item()
                        if score > best_score:
                            best_score = score
                            best_action = action
                            best_idx = i
                        worst_score = min(worst_score, score)
                    except Exception:
                        continue
                self.decisions.append({
                    "action": best_action.action_type.name,
                    "spread": round(best_score - worst_score, 4),
                    "best": round(best_score, 4),
                    "n": len(actions),
                    "chosen_idx": best_idx,
                })
                return best_action

        players = [
            GreedyRLPlayer(colors[0], copy.deepcopy(self.model), self.device),
            WeightedRandomPlayer(colors[1]),
            WeightedRandomPlayer(colors[2]),
            WeightedRandomPlayer(colors[3]),
        ]

        print(f"\n{'='*50}")
        print(f"[Value Net] Game check ep {episode}")
        try:
            game = Game(players)
            winner = game.play()
            red = players[0]
            vp = game.state.player_state.get("P0_VICTORY_POINTS", 0)
            result = "WIN" if winner == colors[0] else "LOSS"
            print(f"  Result: {result} | RED VP: {vp}")

            if red.decisions:
                spreads = [d["spread"] for d in red.decisions]
                avg_spread = sum(spreads) / len(spreads)
                flat = sum(1 for s in spreads if s < 0.001)
                always_first = sum(1 for d in red.decisions if d["chosen_idx"] == 0)
                print(f"  Decisions: {len(red.decisions)} | Avg spread: {avg_spread:.4f} | "
                      f"Flat: {flat} | Chose[0]: {always_first}")
                for d in red.decisions[:3]:
                    print(f"    {d['action']:<22} best={d['best']:.4f} spread={d['spread']:.4f} "
                          f"(#{d['chosen_idx']}/{d['n']})")
                if avg_spread < 0.001:
                    print(f"  FLAT — model cannot distinguish moves")
                elif avg_spread < 0.01:
                    print(f"  WEAK — learning but poor discrimination")
                elif avg_spread < 0.05:
                    print(f"  MODERATE — real preferences emerging")
                else:
                    print(f"  STRONG — clear move discrimination")
        except Exception as e:
            print(f"  Check failed: {e}")
        print(f"{'='*50}\n")

    def train(self, colors=None):
        if colors is None:
            colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        logger.info(f"Outcome-based Value Net: {self.num_episodes} episodes")
        logger.info(f"Feature dim: {get_feature_dim()}")
        logger.info(f"Epsilon: {self.epsilon_start} -> {self.epsilon_end}")
        logger.info(f"Curriculum: Random(0-30%) -> WeightedRandom(30-70%) -> Self-play(70-100%)")

        total_loss = 0.0
        wins = {c: 0 for c in colors}
        best_spread = 0.0

        for episode in range(1, self.num_episodes + 1):
            self._current_episode = episode
            epsilon = self._epsilon(episode)
            episode_data = self._collect_episode(colors, epsilon)

            if episode_data is None:
                continue

            self._push_episode(episode_data)

            for color, data in episode_data.items():
                if data["outcome"] == 1.0:
                    wins[color] += 1

            if episode % self.update_every == 0:
                loss = self._train_step()
                total_loss += loss
                self.losses.append(loss)

            # Snapshot for self-play pool
            if episode % self.snapshot_every == 0:
                snapshot = copy.deepcopy(self.model)
                snapshot.eval()
                self.model_pool.append(snapshot)
                if len(self.model_pool) > self.pool_size:
                    self.model_pool.pop(0)

            if episode % 50 == 0:
                avg_loss = total_loss / max(len(self.losses), 1)
                health = self._health_check()
                progress = episode / self.num_episodes
                if progress < 0.3: phase = "Random"
                elif progress < 0.7: phase = "WeightedRandom"
                else: phase = "Self-play"

                logger.info(
                    f"Ep {episode}/{self.num_episodes} [{phase}] "
                    f"eps={epsilon:.3f} | Buffer: {len(self.buffer)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"pred_std={health.get('pred_std', 0):.4f} "
                    f"({health.get('status', '?')})"
                )
                total_loss = 0.0

            if episode % 200 == 0:
                self._real_game_check(episode, colors)

        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")

        return self.model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default="checkpoints/rl_value/value_network.pt")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    trainer = OutcomeValueTrainer(
        model_path=args.output,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    model = trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
