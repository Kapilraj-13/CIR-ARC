"""
Leakage detection for Phase 1 data splits.
Run: python scripts/check_leakage.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from cir_arc.core.task import ArcTask


def load_all_tasks(directory: Path) -> list:
    tasks = []
    if not directory.exists():
        print(f"  Directory {directory} does not exist, skipping.")
        return tasks
    for json_path in directory.rglob("*.json"):
        try:
            tasks.append(ArcTask.load(json_path))
        except Exception as e:
            print(f"Warning: could not load {json_path}: {e}")
    return tasks


def check_train_held_out_overlap(
    train_dir: Path,
    held_out_dir: Path
) -> int:
    print("Loading train tasks...")
    train_tasks = load_all_tasks(train_dir)
    print(f"  {len(train_tasks)} train tasks loaded")

    print("Loading held-out tasks...")
    held_tasks = load_all_tasks(held_out_dir)
    print(f"  {len(held_tasks)} held-out tasks loaded")

    train_hashes = {t.content_hash for t in train_tasks}
    collisions = 0
    for task in held_tasks:
        if task.content_hash in train_hashes:
            collisions += 1
            print(f"  LEAK DETECTED: {task.task_id} (hash {task.content_hash})")

    return collisions


def check_official_eval_not_in_synthetic(
    official_eval_dir: Path,
    synthetic_dir: Path
) -> int:
    print("Loading official eval tasks...")
    official = load_all_tasks(official_eval_dir)

    print("Loading synthetic tasks...")
    synthetic = load_all_tasks(synthetic_dir)

    official_hashes = {t.content_hash for t in official}
    collisions = 0
    for task in synthetic:
        if task.content_hash in official_hashes:
            collisions += 1
            print(f"  OFFICIAL EVAL LEAK: {task.task_id}")

    return collisions


if __name__ == "__main__":
    base = Path("data")
    total_leaks = 0

    print("\n=== LEAKAGE CHECK 1: Train vs Held-Out ===")
    leaks = check_train_held_out_overlap(
        base / "synthetic" / "train",
        base / "synthetic" / "held_out"
    )
    print(f"Result: {leaks} collisions\n")
    total_leaks += leaks

    print("=== LEAKAGE CHECK 2: Official Eval in Synthetic ===")
    official_eval = base / "official" / "evaluation"
    if official_eval.exists():
        leaks = check_official_eval_not_in_synthetic(
            official_eval,
            base / "synthetic"
        )
        print(f"Result: {leaks} collisions\n")
        total_leaks += leaks
    else:
        print(f"  {official_eval} not found, skipping.\n")

    if total_leaks == 0:
        print("OK: No leakage detected. Data splits are clean.")
    else:
        print(f"FAIL: {total_leaks} leaks detected. Regenerate data with new seeds.")
        exit(1)
