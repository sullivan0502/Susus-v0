"""
Necto Brain Adapter for susu_bot (Susu-v0).
Wraps necto-model.pt (EARL Perceiver MultiheadAttention) and NectoObsBuilder.
Produces 8-float action vectors matching the RLBot standard controls.
"""

import math
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from necto_obs import NectoObsBuilder


class NectoBrain:
    """
    Wraps necto-model.pt (EARL Perceiver) for 1v1 match play and kickoffs.
    """

    def __init__(self, field_info=None):
        cur_dir = Path(__file__).resolve().parent
        model_path = cur_dir / "necto-model.pt"

        self.actor = None
        self.obs_builder: Optional[NectoObsBuilder] = None
        self.device = "cpu"

        if model_path.exists():
            try:
                self.actor = torch.jit.load(str(model_path), map_location=self.device)
                self.actor.eval()
                torch.set_num_threads(1)
                self.obs_builder = NectoObsBuilder(field_info=field_info)
                print(f"[NectoBrain] Moteur EARL Perceiver charge avec succes : {model_path.name}")
            except Exception as e:
                print(f"[NectoBrain] Erreur chargement {model_path.name}: {e}")
                self.actor = None
        else:
            print(f"[NectoBrain] necto-model.pt introuvable dans {cur_dir}")

    @property
    def is_ready(self) -> bool:
        return self.actor is not None and self.obs_builder is not None

    def initialize(self, field_info):
        self.obs_builder = NectoObsBuilder(field_info=field_info)

    def get_action_and_weights(
        self, player, game_state, previous_action: np.ndarray, beta: float = 1.0
    ) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor]]:
        if not self.is_ready:
            return None, None

        try:
            obs = self.obs_builder.build_obs(player, game_state, previous_action)
            state = tuple(torch.from_numpy(s).float() for s in obs)

            with torch.no_grad():
                out, weights = self.actor(state)

            max_shape = max(o.shape[-1] for o in out)
            logits = torch.stack(
                [
                    l
                    if l.shape[-1] == max_shape
                    else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf") if beta >= 0 else float("inf"))
                    for l in out
                ]
            ).swapdims(0, 1).squeeze()

            if beta == 1.0:
                actions = np.argmax(logits, axis=-1)
            elif beta == -1.0:
                actions = np.argmin(logits, axis=-1)
            else:
                if beta == 0.0:
                    logits[torch.isfinite(logits)] = 0.0
                else:
                    scale = math.log((beta + 1.0) / (1.0 - beta), 3.0)
                    logits = logits * scale
                dist = Categorical(logits=logits)
                actions = dist.sample().numpy()

            actions = actions.reshape((-1, 5))
            actions[:, 0] = actions[:, 0] - 1
            actions[:, 1] = actions[:, 1] - 1

            parsed = np.zeros((actions.shape[0], 8), dtype=np.float32)
            parsed[:, 0] = actions[:, 0]  # throttle
            parsed[:, 1] = actions[:, 1]  # steer
            parsed[:, 2] = actions[:, 0]  # pitch
            parsed[:, 3] = actions[:, 1] * (1 - actions[:, 4])  # yaw
            parsed[:, 4] = actions[:, 1] * actions[:, 4]  # roll
            parsed[:, 5] = actions[:, 2]  # jump
            parsed[:, 6] = actions[:, 3]  # boost
            parsed[:, 7] = actions[:, 4]  # handbrake

            return parsed[0].copy(), weights
        except Exception as e:
            return None, None
