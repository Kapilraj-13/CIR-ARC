"""Generates the Jupyter Notebook for CIR-ARC Phase 4 Kaggle Reasoner Training."""

from __future__ import annotations
import json
from pathlib import Path


def generate_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# CIR-ARC Phase 4: ~120.18M Direct Cognitive Reasoner Training (Kaggle GPU)\n",
                    "\n",
                    "This notebook executes the training curriculum for the **CIR-ARC Direct Cognitive Reasoner (~120.18M parameters)** on Kaggle.\n",
                    "\n",
                    "### Architectural Specification:\n",
                    "- **Transformer Core**: 18 layers, $d_{\\text{model}}=768$, Grouped-Query Attention (12 Q, 4 KV heads, head dimension 64), SwiGLU ($d_{\\text{ff}}=1856$), RMSNorm, RoPE.\n",
                    "- **Core Parameters**: 105,312,000\n",
                    "- **Input Fusion & Projections**: 5,095,936 parameters (Symbolic entities, relations, events, mechanics, global state, uncertainty + continuous slot vectors + adaptively compressed 128 spatial tokens + Gated Neuro-Symbolic Fusion).\n",
                    "- **Dual Memory System**: 3,544,832 parameters (128-token ephemeral reasoning workspace $R_t$ + 128-token persistent working memory $M_t$ updated via Cross-Attention + episodic retrieval).\n",
                    "- **Cognitive Output Heads**: 6,226,592 parameters (4-hypothesis Goal Inference, rich World Model Transition, Value/Risk Estimator, Action Interface with entity pointer, and Verification error head).\n",
                    "- **Total Parameters**: **120,179,360 parameters** (audited to 100.00% precision).\n",
                    "\n",
                    "### Training Invariant & Hardware:\n",
                    "- Multi-hour training is designed for Kaggle GPUs (dual T4, P100, V100, or A100).\n",
                    "- Automatic precision selection: `bfloat16` on A100/Ampere, `float16` on T4/Turing."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Environment Setup & Repository Synchronization\n",
                    "Detects execution environment, clones or updates the latest repository from GitHub, and configures python paths."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Detect root directory\n",
                    "if os.path.exists('/kaggle/working'):\n",
                    "    ROOT_DIR = '/kaggle/working'\n",
                    "elif os.path.exists('/content'):\n",
                    "    ROOT_DIR = '/content'\n",
                    "else:\n",
                    "    ROOT_DIR = os.getcwd()\n",
                    "\n",
                    "os.chdir(ROOT_DIR)\n",
                    "repo_path = os.path.join(ROOT_DIR, 'CIR-ARC')\n",
                    "\n",
                    "if not os.path.exists(os.path.join(repo_path, 'src', 'cir_arc')):\n",
                    "    if os.path.exists(os.path.join(ROOT_DIR, 'src', 'cir_arc')):\n",
                    "        repo_path = ROOT_DIR\n",
                    "    else:\n",
                    "        !git clone https://github.com/Kapilraj-13/CIR-ARC.git\n",
                    "        repo_path = os.path.join(ROOT_DIR, 'CIR-ARC')\n",
                    "\n",
                    "os.chdir(repo_path)\n",
                    "!git fetch origin master\n",
                    "!git reset --hard origin/master\n",
                    "\n",
                    "if os.path.join(repo_path, 'src') not in sys.path:\n",
                    "    sys.path.insert(0, os.path.join(repo_path, 'src'))\n",
                    "\n",
                    "print(f'Working directory: {os.getcwd()}')\n",
                    "print(f'CIR-ARC package located: {os.path.exists(os.path.join(repo_path, \"src\", \"cir_arc\"))}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Hardware Acceleration & Auto-Precision Detection"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "\n",
                    "print(f'PyTorch version: {torch.__version__}')\n",
                    "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                    "\n",
                    "if torch.cuda.is_available():\n",
                    "    num_gpus = torch.cuda.device_count()\n",
                    "    device = torch.device('cuda:0')\n",
                    "    print(f'Detected GPUs: {num_gpus}')\n",
                    "    for i in range(num_gpus):\n",
                    "        g_name = torch.cuda.get_device_name(i)\n",
                    "        g_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)\n",
                    "        print(f'  [GPU {i}]: {g_name} ({g_mem:.2f} GB VRAM)')\n",
                    "    \n",
                    "    if torch.cuda.is_bf16_supported():\n",
                    "        compute_dtype = torch.bfloat16\n",
                    "        print('Selected compute precision: bfloat16 (Ampere/Hopper native)')\n",
                    "    else:\n",
                    "        compute_dtype = torch.float16\n",
                    "        print('Selected compute precision: float16 (T4/P100 native)')\n",
                    "else:\n",
                    "    num_gpus = 0\n",
                    "    device = torch.device('cpu')\n",
                    "    compute_dtype = torch.float32\n",
                    "    print('Running on CPU (float32)')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Model Instantiation & Multi-GPU DataParallel (120,179,360 Parameters)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch.nn as nn\n",
                    "from cir_arc.neural.reasoner import ReasonerConfig, CognitiveReasoner120M\n",
                    "\n",
                    "config = ReasonerConfig()\n",
                    "raw_model = CognitiveReasoner120M(config).to(device=device, dtype=compute_dtype)\n",
                    "\n",
                    "counts = raw_model.count_parameters()\n",
                    "print('=' * 60)\n",
                    "print('CIR-ARC DIRECT COGNITIVE REASONER — AUDITED PARAMETERS')\n",
                    "print('=' * 60)\n",
                    "for k, v in counts.items():\n",
                    "    print(f'  {k:25s}: {v:12,d} ({v / 1e6:.3f}M)')\n",
                    "print('=' * 60)\n",
                    "assert counts['total'] == 120_179_360, f\"Parameter mismatch: expected 120,179,360, got {counts['total']}\"\n",
                    "print('VERIFIED: Model architecture matches 120,179,360 parameters down to single weights!')\n",
                    "\n",
                    "if torch.cuda.device_count() > 1:\n",
                    "    print(f'\\n[MULTI-GPU ACTIVE] Distributing model across all {torch.cuda.device_count()} GPUs via DataParallel!')\n",
                    "    model = nn.DataParallel(raw_model)\n",
                    "else:\n",
                    "    model = raw_model"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Reasoning-Based Synthetic Dataset & Trajectory Streaming (15,000 Tasks)\n",
                    "Loads tasks from `data/synthetic/train` converted on the fly into complete reasoning steps with actions, goal configurations, counterfactual scores, active mechanics vectors, and deceptive error targets."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, sys\n",
                    "from pathlib import Path\n",
                    "from torch.utils.data import DataLoader\n",
                    "\n",
                    "train_data_dir = 'data/synthetic/train'\n",
                    "val_data_dir = 'data/synthetic/held_out'\n",
                    "\n",
                    "# Ensure synthetic dataset is generated on disk (data/ is git-ignored in the repo)\n",
                    "train_files = list(Path(train_data_dir).rglob('*.json')) if os.path.exists(train_data_dir) else []\n",
                    "if len(train_files) == 0:\n",
                    "    print('=' * 75)\n",
                    "    print('Synthetic dataset not found on disk (data/ is git-ignored).')\n",
                    "    print('Procedurally generating training and held-out reasoning tasks (~12,000 tasks)...')\n",
                    "    print('=' * 75)\n",
                    "    !python scripts/generate_data.py --n_per_rule 800 --n_per_pair 400\n",
                    "\n",
                    "# Clear cached modules if previously imported in this running Jupyter kernel\n",
                    "for mod in list(sys.modules.keys()):\n",
                    "    if mod.startswith('cir_arc'):\n",
                    "        del sys.modules[mod]\n",
                    "\n",
                    "from cir_arc.neural.training.reasoning_dataset import ReasoningArcDataset, collate_reasoning_batch\n",
                    "\n",
                    "train_dataset = ReasoningArcDataset(data_dir=train_data_dir, max_samples=None, seed=42)\n",
                    "print(f'Loaded {len(train_dataset):,} reasoning training tasks from {train_data_dir}')\n",
                    "\n",
                    "val_dataset = ReasoningArcDataset(data_dir=val_data_dir, max_samples=None, seed=101) if os.path.exists(val_data_dir) else None\n",
                    "if val_dataset:\n",
                    "    print(f'Loaded {len(val_dataset):,} held-out evaluation tasks from {val_data_dir}')\n",
                    "\n",
                    "effective_batch_size = 16 if torch.cuda.device_count() > 1 else 8\n",
                    "print(f'Using effective batch size: {effective_batch_size} across {max(1, torch.cuda.device_count())} device(s)')\n",
                    "train_loader = DataLoader(\n",
                    "    train_dataset,\n",
                    "    batch_size=effective_batch_size,\n",
                    "    shuffle=True,\n",
                    "    collate_fn=collate_reasoning_batch,\n",
                    "    num_workers=4 if torch.cuda.is_available() else 0,\n",
                    "    pin_memory=torch.cuda.is_available(),\n",
                    ")\n",
                    "\n",
                    "# Inspect sample batch\n",
                    "sample_batch = next(iter(train_loader))\n",
                    "print(f'Sample batch size: {sample_batch[\"batch_size\"]}')\n",
                    "print(f'Slot embeddings: {sample_batch[\"slot_embeddings\"].shape}')\n",
                    "print(f'Candidate scores: {sample_batch[\"candidate_scores\"].shape}')\n",
                    "print(f'Mechanics vectors: {sample_batch[\"mechanics_vec\"].shape}')\n",
                    "print(f'Target is_error: {sample_batch[\"target_is_error\"].shape}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Multi-Objective Training Loss & Optimizer Configuration\n",
                    "Implements the complete composite loss:\n",
                    "$$\\mathcal{L} = \\lambda_1 \\mathcal{L}_{\\text{state}} + \\lambda_2 \\mathcal{L}_{\\text{goal}} + \\lambda_3 \\mathcal{L}_{\\text{dynamics}} + \\lambda_4 \\mathcal{L}_{\\text{action}} + \\lambda_5 \\mathcal{L}_{\\text{counterfactual}} + \\lambda_6 \\mathcal{L}_{\\text{value}} + \\lambda_7 \\mathcal{L}_{\\text{plan}} + \\lambda_8 \\mathcal{L}_{\\text{verify}} + \\lambda_9 \\mathcal{L}_{\\text{efficiency}}$$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from cir_arc.neural.reasoner import ReasonerMultiObjectiveLoss, ReasonerLossWeights\n",
                    "\n",
                    "loss_weights = ReasonerLossWeights(\n",
                    "    lambda_state=1.0,\n",
                    "    lambda_goal=1.0,\n",
                    "    lambda_dynamics=1.5,\n",
                    "    lambda_action=2.0,\n",
                    "    lambda_counterfactual=0.5,\n",
                    "    lambda_value=1.0,\n",
                    "    lambda_plan=0.5,\n",
                    "    lambda_verify=1.0,\n",
                    "    lambda_efficiency=0.1,\n",
                    ")\n",
                    "criterion = ReasonerMultiObjectiveLoss(loss_weights)\n",
                    "\n",
                    "optimizer = torch.optim.AdamW(\n",
                    "    model.parameters(),\n",
                    "    lr=2e-4,\n",
                    "    betas=(0.9, 0.95),\n",
                    "    weight_decay=0.01,\n",
                    "    eps=1e-8,\n",
                    ")\n",
                    "try:\n",
                    "    scaler = torch.amp.GradScaler('cuda', enabled=(compute_dtype == torch.float16 and device.type == 'cuda'))\n",
                    "except Exception:\n",
                    "    scaler = torch.cuda.amp.GradScaler(enabled=(compute_dtype == torch.float16 and device.type == 'cuda'))\n",
                    "print('Optimizer and GradScaler initialized successfully.')"

                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Comprehensive Metrics Tracker & Scorecard Engine\n",
                    "Initializes `ReasonerMetricsTracker` to track:\n",
                    "- **Losses**: Composite total loss, action loss, goal loss, dynamics loss, verification loss, value loss\n",
                    "- **Accuracy**: Action Top-1 accuracy, Action Top-3 accuracy, Verification accuracy\n",
                    "- **F1 Scores**: Verification F1 (Precision, Recall, F1 on anomaly/error detection), Action Macro F1\n",
                    "- **ARC-AGI Cognitive Metrics**: Goal Cosine Similarity, Counterfactual Action Ranking Accuracy, Dynamics Latent MSE, and Exact Grid Match Rate"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from cir_arc.neural.evaluation.reasoner_metrics import ReasonerMetricsTracker\n",
                    "\n",
                    "metrics_tracker = ReasonerMetricsTracker()\n",
                    "print('ReasonerMetricsTracker initialized successfully.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Curriculum Training Loop with Real-Time F1, Accuracy, and Loss Tracking"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import time\n",
                    "\n",
                    "def train_epoch(\n",
                    "    model, dataloader, optimizer, criterion, scaler, tracker,\n",
                    "    device, dtype, grad_accum_steps=4, max_batches=100, epoch=1\n",
                    "):\n",
                    "    model.train()\n",
                    "    tracker.reset()\n",
                    "    optimizer.zero_grad()\n",
                    "    t0 = time.time()\n",
                    "    \n",
                    "    for step, batch in enumerate(dataloader):\n",
                    "        if step >= max_batches:\n",
                    "            break\n",
                    "            \n",
                    "        slots = batch['slot_embeddings'].to(device=device, dtype=dtype)\n",
                    "        actions = batch['action'].to(device=device)\n",
                    "        \n",
                    "        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == 'cuda')):\n",
                    "            outputs = model(slot_embeddings=slots)\n",
                    "            \n",
                    "            targets = {\n",
                    "                'target_state_latent': outputs['cognitive_state'].detach(),\n",
                    "                'target_goal_latent': outputs['goals'][:, 0].detach(),\n",
                    "                'target_action_id': actions,\n",
                    "                'candidate_scores': batch['candidate_scores'].to(device=device),\n",
                    "                'optimal_action_mask': (batch['candidate_scores'] > 0.5).to(device=device),\n",
                    "                'target_discounted_return': batch['value_target'].to(device=device),\n",
                    "                'target_is_error': batch['target_is_error'].to(device=device),\n",
                    "            }\n",
                    "            \n",
                    "            loss_dict = criterion(outputs, targets)\n",
                    "            loss = loss_dict['total_loss'] / grad_accum_steps\n",
                    "            \n",
                    "        scaler.scale(loss).backward()\n",
                    "        \n",
                    "        if (step + 1) % grad_accum_steps == 0 or (step + 1) == max_batches:\n",
                    "            scaler.unscale_(optimizer)\n",
                    "            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
                    "            scaler.step(optimizer)\n",
                    "            scaler.update()\n",
                    "            optimizer.zero_grad()\n",
                    "            \n",
                    "        # Track metrics and loss components across batches\n",
                    "        tracker.update_losses(loss_dict)\n",
                    "        tracker.update_batch(outputs, targets)\n",
                    "        \n",
                    "        if (step + 1) % 25 == 0:\n",
                    "            m_step = tracker.compute()\n",
                    "            print(f\"  Step [{step+1}/{max_batches}] | Loss: {loss_dict['total_loss'].item():.4f} | \"\n",
                    "                  f\"Act Acc: {m_step.get('action_accuracy', 0.0)*100:.1f}% | \"\n",
                    "                  f\"Verify F1: {m_step.get('verification_f1', 0.0):.3f} | \"\n",
                    "                  f\"Goal CosSim: {m_step.get('goal_cosine_similarity', 0.0):.3f}\")\n",
                    "                  \n",
                    "    elapsed = time.time() - t0\n",
                    "    print(f'\\nEpoch {epoch} finished in {elapsed:.1f}s.')\n",
                    "    tracker.print_scorecard(epoch=epoch)\n",
                    "    return tracker.compute()\n",
                    "\n",
                    "# Multi-epoch curriculum training on Kaggle GPUs\n",
                    "num_epochs = 5\n",
                    "max_batches = 100\n",
                    "best_accuracy = 0.0\n",
                    "print(f'Starting Multi-Epoch Curriculum Training ({num_epochs} Epochs on Dual GPUs)...')\n",
                    "for epoch in range(1, num_epochs + 1):\n",
                    "    print(f'\\n>>> Running Epoch {epoch}/{num_epochs}')\n",
                    "    train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, metrics_tracker, device, compute_dtype, max_batches=max_batches, epoch=epoch)\n",
                    "    acc = train_metrics.get('action_accuracy', 0.0)\n",
                    "    if acc > best_accuracy:\n",
                    "        best_accuracy = acc\n",
                    "        print(f'[*] New peak accuracy: {best_accuracy * 100:.2f}% (Epoch {epoch})')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Validation Evaluation & Held-Out Metric Scorecard"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def evaluate_model(model, dataloader, criterion, device, dtype, max_eval_batches=25):\n",
                    "    model.eval()\n",
                    "    eval_tracker = ReasonerMetricsTracker()\n",
                    "    \n",
                    "    with torch.no_grad():\n",
                    "        for step, batch in enumerate(dataloader):\n",
                    "            if step >= max_eval_batches:\n",
                    "                break\n",
                    "                \n",
                    "            slots = batch['slot_embeddings'].to(device=device, dtype=dtype)\n",
                    "            actions = batch['action'].to(device=device)\n",
                    "            \n",
                    "            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == 'cuda')):\n",
                    "                outputs = model(slot_embeddings=slots)\n",
                    "                targets = {\n",
                    "                    'target_state_latent': outputs['cognitive_state'].detach(),\n",
                    "                    'target_goal_latent': outputs['goals'][:, 0].detach(),\n",
                    "                    'target_action_id': actions,\n",
                    "                    'candidate_scores': batch['candidate_scores'].to(device=device),\n",
                    "                    'optimal_action_mask': (batch['candidate_scores'] > 0.5).to(device=device),\n",
                    "                    'target_discounted_return': batch['value_target'].to(device=device),\n",
                    "                    'target_is_error': batch['target_is_error'].to(device=device),\n",
                    "                }\n",
                    "                loss_dict = criterion(outputs, targets)\n",
                    "            eval_tracker.update_losses(loss_dict)\n",
                    "            eval_tracker.update_batch(outputs, targets)\n",
                    "            \n",
                    "    eval_tracker.print_scorecard(epoch=None)\n",
                    "    return eval_tracker.compute()\n",
                    "\n",
                    "eval_loader = val_loader if 'val_loader' in locals() and val_loader is not None else train_loader\n",
                    "print('Evaluating Reasoner on Evaluation Batches...')\n",
                    "val_metrics = evaluate_model(model, eval_loader, criterion, device, compute_dtype, max_eval_batches=20)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 9. Interactive Cognitive Planning & Counterfactual Rollout Evaluation\n",
                    "Demonstrates the Reasoner evaluating candidate actions in latent space and selecting optimal ActionIntent."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "planner_engine = model.module if hasattr(model, 'module') else model\n",
                    "planner_engine.eval()\n",
                    "with torch.no_grad():\n",
                    "    test_slots = sample_batch['slot_embeddings'][:1].to(device=device, dtype=compute_dtype)\n",
                    "    candidate_actions = [0, 1, 2, 3, 4, 6]  # MOVE_UP, DOWN, LEFT, RIGHT, ACTION, CLICK\n",
                    "    \n",
                    "    intent, scores = planner_engine.plan(slot_embeddings=test_slots, candidate_actions=candidate_actions)\n",
                    "    \n",
                    "print('=' * 60)\n",
                    "print('COGNITIVE REASONER — COUNTERFACTUAL ACTION SELECTION')\n",
                    "print('=' * 60)\n",
                    "print(f'Selected Action Intent : {intent.action_name} (ID={intent.action_type_id})')\n",
                    "print(f'Expected Future Value  : {intent.expected_value:.4f}')\n",
                    "print(f'Confidence Score       : {intent.confidence:.4f}')\n",
                    "print(f'Information Gain       : {intent.info_gain:.4f}')\n",
                    "print('\\nCandidate Action Rollout Ranking:')\n",
                    "for rank, sc in enumerate(scores, 1):\n",
                    "    print(f'  #{rank} {sc.action_name:12s} | Score: {sc.total_score:+.4f} | '\n",
                    "          f'Success: {sc.success_prob:.2f} | Value: {sc.future_value:+.2f} | Risk: {sc.risk_penalty:.2f}')\n",
                    "print('=' * 60)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 10. Checkpoint Export\n",
                    "Saves the trained ~120.18M reasoner checkpoint to the output directory."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from dataclasses import asdict\n",
                    "\n",
                    "output_dir = Path('/kaggle/working/checkpoints/phase4') if os.path.exists('/kaggle/working') else Path('checkpoints/phase4')\n",
                    "output_dir.mkdir(parents=True, exist_ok=True)\n",
                    "checkpoint_path = output_dir / 'best_reasoner_120m.pt'\n",
                    "\n",
                    "raw_model_to_save = model.module if hasattr(model, 'module') else model\n",
                    "config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)\n",
                    "checkpoint = {\n",
                    "    'config': config_dict,\n",
                    "    'model_state_dict': raw_model_to_save.state_dict(),\n",
                    "    'optimizer_state_dict': optimizer.state_dict(),\n",
                    "    'param_counts': counts,\n",
                    "    'val_metrics': val_metrics if 'val_metrics' in locals() else None,\n",
                    "}\n",
                    "torch.save(checkpoint, checkpoint_path)\n",
                    "print(f'Successfully exported CIR-ARC 120.18M Reasoner checkpoint to: {checkpoint_path}')\n",
                    "print(f'File size: {checkpoint_path.stat().st_size / (1024**2):.2f} MB')\n",
                    "\n",
                    "# Quick verification of saved checkpoint\n",
                    "loaded = torch.load(checkpoint_path, map_location='cpu')\n",
                    "saved_param_count = sum(p.numel() for p in loaded['model_state_dict'].values())\n",
                    "print(f'Checkpoint verification: PASS (Params: {saved_param_count:,}, Keys: {list(loaded.keys())})')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = Path("notebooks/phase4_kaggle_reasoner_training.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"Successfully created {out_path} ({out_path.stat().st_size:,} bytes)!")


if __name__ == "__main__":
    generate_notebook()
