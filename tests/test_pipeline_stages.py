import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adafactor

# Add src to path
sys.path.insert(0, os.path.abspath('src'))
from cir_arc.neural.models.cir_arc_3b import CirArc3B, CirArc3BConfig

print("[TEST] Initializing CIR-ARC-3B on CPU for Pipeline Test...")
config = CirArc3BConfig()
base_model = CirArc3B(config)

# Stage 1: Perception + Token Interface + Trunk Blocks 0-11
class CirArc3BStage1(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.perception = m.perception
        self.token_interface = m.token_interface
        self.layers = nn.ModuleList([m.trunk.layers[i] for i in range(12)])
        self.config = m.config

    def forward(self, grid_t, grid_next=None, input_token_ids=None):
        B = grid_t.shape[0]
        perc_out = self.perception(grid_t, grid_next)
        slot_tokens = perc_out['trunk_tokens']
        slots = perc_out['slots']
        if input_token_ids is not None:
            text_tokens = self.token_interface.embed_tokens(input_token_ids)
            trunk_inputs = torch.cat([slot_tokens, text_tokens], dim=1)
        else:
            trunk_inputs = slot_tokens
        h = trunk_inputs
        for layer in self.layers:
            h, _ = layer(h)
        return h, slots

# Stage 2: Trunk Blocks 12-23 + Cognitive Faculties
class CirArc3BStage2(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.layers = nn.ModuleList([m.trunk.layers[i] for i in range(12, 24)])
        self.final_norm = m.token_interface.final_norm
        self.game_model = m.game_model
        self.causal_graph = m.causal_graph
        self.world_model = m.world_model
        self.hypothesis_engine = m.hypothesis_engine
        self.memory_falsifier = m.memory_falsifier
        self.mpc_planner = m.mpc_planner
        self.action_heads = m.action_heads
        self.config = m.config

    def forward(self, h, slots, action=None, event_stream=None):
        for layer in self.layers:
            h, _ = layer(h)
        h = self.final_norm(h)
        cognitive_state = h[:, 0, :]
        game_res = self.game_model(slots, cognitive_state, action)
        reversibility = game_res['reversibility']
        causal_res = self.causal_graph(slots, cognitive_state)
        act_wm = action if action is not None else torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        wm_res = self.world_model(cognitive_state, act_wm, cognitive_state)
        hypo_res = self.hypothesis_engine(cognitive_state, wm_res['predicted_state'])
        active_hypo = hypo_res['hypotheses'][:, 0, :]
        if event_stream is None:
            event_stream = cognitive_state.unsqueeze(1).expand(-1, 16, -1)
        mem_res = self.memory_falsifier(cognitive_state, event_stream, active_hypo)
        mpc_res = self.mpc_planner(cognitive_state, active_hypo)
        action_res = self.action_heads(cognitive_state, reversibility_score=reversibility)
        return {
            'cognitive_state': cognitive_state,
            'world_model': wm_res,
            'policy_logits': action_res['policy_logits'],
            'entrapment_risk': action_res['entrapment_risk']
        }

stage1 = CirArc3BStage1(base_model)
stage2 = CirArc3BStage2(base_model)

# Deduplicate trainable parameters
trainable_params = []
seen = set()
for name, p in base_model.named_parameters():
    if 'calibrated_weights' in name:
        p.requires_grad = False
    elif id(p) not in seen:
        seen.add(id(p))
        trainable_params.append(p)

print(f"[TEST] Extracted {len(trainable_params)} unique active trainable parameters (expected 390).")
assert len(trainable_params) == 390, f"Expected 390, got {len(trainable_params)}"

optimizer = Adafactor(trainable_params, lr=2e-4, weight_decay=0.1)

def compute_loss(model_out, act, neg):
    policy_logits = model_out['policy_logits'].float()
    l_policy = F.cross_entropy(policy_logits, act.to(policy_logits.device))

    entrapment_risk = model_out['entrapment_risk'].squeeze(-1).float()
    entrapment_risk = torch.nan_to_num(entrapment_risk, nan=0.5, posinf=0.9999, neginf=0.0001)
    entrapment_risk = entrapment_risk.clamp(1e-6, 1.0 - 1e-6)
    target_neg = torch.nan_to_num(neg.to(entrapment_risk.device).float(), nan=0.0).clamp(0.0, 1.0)
    l_safety = F.binary_cross_entropy(entrapment_risk, target_neg)

    pred_state = model_out['world_model']['predicted_state'].float()
    cog_state = model_out['cognitive_state'].detach().float()
    l_dynamics = F.mse_loss(pred_state, cog_state)

    total = l_policy + 0.5 * l_safety + 0.25 * l_dynamics
    return total, l_policy, l_safety

print("[TEST] Running 1 mock step with dummy tensors...")
B = 2
gt_A = torch.zeros(B, 11, 32, 32)
gn_A = torch.zeros(B, 11, 32, 32)
act_A = torch.zeros(B, dtype=torch.long)
neg_A = torch.zeros(B, dtype=torch.float32)

gt_B = torch.zeros(B, 11, 32, 32)
gn_B = torch.zeros(B, 11, 32, 32)
act_B = torch.zeros(B, dtype=torch.long)
neg_B = torch.zeros(B, dtype=torch.float32)

optimizer.zero_grad()
h_A, slots_A = stage1(gt_A, gn_A)
out_A = stage2(h_A, slots_A, act_A)
loss_A, p_loss_A, s_loss_A = compute_loss(out_A, act_A, neg_A)
loss_A = loss_A / 2
loss_A.backward()

h_B, slots_B = stage1(gt_B, gn_B)
out_B = stage2(h_B, slots_B, act_B)
loss_B, p_loss_B, s_loss_B = compute_loss(out_B, act_B, neg_B)
loss_B = loss_B / 2
loss_B.backward()

torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
optimizer.step()
optimizer.zero_grad()

print(f"[TEST PASSED] Step completed! Loss A: {loss_A.item():.4f}, Loss B: {loss_B.item():.4f}")
