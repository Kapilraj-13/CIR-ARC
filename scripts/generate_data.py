"""
Phase 1 dataset generation script.
Usage: python scripts/generate_data.py --n_per_rule 1000
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from cir_arc.generators.single_rule import GENERATOR_REGISTRY
from cir_arc.generators.composition import TwoRuleGenerator

TRAIN_SEED = 42
HELD_OUT_SEED = 200
SPLIT_RATIO = 0.8

COMPOSITION_PAIRS = [
    ("reflect_horizontal", "color_swap_all"),
    ("rotate_90", "color_swap_all"),
    ("gravity", "reflect_vertical"),
    ("scale_up", "color_swap_all"),
]


def generate_single_rule_data(n_per_rule: int, output_dir: Path):
    for rule_name, GeneratorClass in tqdm(
        GENERATOR_REGISTRY.items(), desc="Single-rule generators"
    ):
        gen = GeneratorClass()
        n_train = int(n_per_rule * SPLIT_RATIO)
        n_held = n_per_rule - n_train

        train_tasks = gen.generate_batch(n_train, seed=TRAIN_SEED)
        train_dir = output_dir / "train" / rule_name
        train_dir.mkdir(parents=True, exist_ok=True)
        for task in train_tasks:
            task.save(train_dir / f"{task.task_id}.json")

        held_tasks = gen.generate_batch(n_held, seed=HELD_OUT_SEED)
        held_dir = output_dir / "held_out" / rule_name
        held_dir.mkdir(parents=True, exist_ok=True)
        for task in held_tasks:
            task.save(held_dir / f"{task.task_id}.json")

        print(f"  {rule_name}: {n_train} train, {n_held} held_out")


def generate_composition_data(n_per_pair: int, output_dir: Path):
    for rule_a, rule_b in tqdm(COMPOSITION_PAIRS, desc="Composition generators"):
        gen = TwoRuleGenerator(rule_a, rule_b)
        n_train = int(n_per_pair * SPLIT_RATIO)
        n_held = n_per_pair - n_train

        train_tasks = gen.generate_batch(n_train, seed=TRAIN_SEED + 1)
        rule_key = f"compose_{rule_a}__{rule_b}"
        train_dir = output_dir / "train" / rule_key
        train_dir.mkdir(parents=True, exist_ok=True)
        for task in train_tasks:
            task.save(train_dir / f"{task.task_id}.json")

        held_tasks = gen.generate_batch(n_held, seed=HELD_OUT_SEED + 1)
        held_dir = output_dir / "held_out" / rule_key
        held_dir.mkdir(parents=True, exist_ok=True)
        for task in held_tasks:
            task.save(held_dir / f"{task.task_id}.json")

        print(f"  {rule_key}: {n_train} train, {n_held} held_out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_rule", type=int, default=1000)
    parser.add_argument("--n_per_pair", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data/synthetic")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Generating to {output_dir}")

    generate_single_rule_data(args.n_per_rule, output_dir)
    generate_composition_data(args.n_per_pair, output_dir)

    print("\nDone. Run leakage check to verify no cross-split contamination.")
