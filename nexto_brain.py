"""
Nexto Brain Adapter for susu_bot (Susu-v0).
Enables borrowing Nexto's Transformer-based EARL (Entity-Aware RL) policy
for high aerials, aerial contests, and precision aerial mechanics.
"""

import math
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from nexto_obs import NextoObsBuilder
from susu_obs import get_lookup_table


class NextoBrain:
    """
    Wraps nexto-model.pt (EARL Transformer) for 1v1/2v2/3v3 inference.
    Outputs logits over the exact same 90-action discrete lookup table as Susu.
    """

    def __init__(self, field_info=None):
        cur_dir = Path(__file__).resolve().parent
        model_path = cur_dir / "nexto-model.pt"

        self.actor = None
        self.obs_builder: Optional[NextoObsBuilder] = None
        self._lookup_table = get_lookup_table()
        self.device = "cpu"

        if model_path.exists():
            try:
                self.actor = torch.jit.load(str(model_path), map_location=self.device)
                self.actor.eval()
                torch.set_num_threads(1)
                self.obs_builder = NextoObsBuilder(field_info=field_info)
                print(f"[NextoBrain] Moteur EARL Transformer charge avec succes : {model_path.name}")
            except Exception as e:
                print(f"[NextoBrain] Avertissement: Impossible de charger {model_path.name}: {e}")
                self.actor = None
        else:
            print(f"[NextoBrain] nexto-model.pt introuvable dans {cur_dir}")

    @property
    def is_ready(self) -> bool:
        return self.actor is not None and self.obs_builder is not None

    def initialize(self, field_info):
        self.obs_builder = NextoObsBuilder(field_info=field_info)

    def get_action_and_logits(
        self, player, game_state, previous_action: np.ndarray, beta: float = 1.0
    ) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor]]:
        if not self.is_ready:
            return None, None

        try:
            obs = self.obs_builder.build_obs(player, game_state, previous_action)
            state = tuple(torch.from_numpy(s).float() for s in obs)

            with torch.no_grad():
                out, weights = self.actor(state)

            out = (out,)
            max_shape = max(o.shape[-1] for o in out)
            logits = torch.stack(
                [
                    l if l.shape[-1] == max_shape
                    else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                    for l in out
                ],
                dim=1,
            ).squeeze(0).squeeze(0)

            if beta == 1.0:
                action_idx = int(torch.argmax(logits).item())
            elif beta == -1.0:
                action_idx = int(torch.argmin(logits).item())
            else:
                if beta == 0.0:
                    logits[torch.isfinite(logits)] = 0.0
                else:
                    scale = math.log((beta + 1.0) / (1.0 - beta), 3.0)
                    logits = logits * scale
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = int(dist.sample().item())

            action = self._lookup_table[action_idx].copy()
            return action, logits
        except Exception as e:
            return None, None
