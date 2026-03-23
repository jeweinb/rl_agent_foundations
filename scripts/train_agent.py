#!/usr/bin/env python3
"""CLI script to train the RL agent: BC warm-start → Actor-Critic CQL fine-tuning."""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHECKPOINTS_DIR
from training.data_loader import load_datasets, build_offline_episodes
from training.behavior_cloning import train_behavior_cloning
from training.cql_trainer import train_cql


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train RL agent (BC → Actor-Critic CQL)")
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--cql-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-bc", action="store_true", help="Skip BC, load existing checkpoint")
    args = parser.parse_args()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # Load data
    print("Loading datasets...")
    datasets = load_datasets()
    print("Building offline episodes...")
    episodes = build_offline_episodes(
        datasets["state_features"],
        datasets["historical_activity"],
        datasets["action_eligibility"],
    )
    print(f"  {len(episodes)} episodes built\n")

    # Phase 1: Behavior Cloning
    bc_path = os.path.join(CHECKPOINTS_DIR, "bc_policy.pt")
    if args.skip_bc and os.path.exists(bc_path):
        from training.behavior_cloning import ActionMaskedPolicy
        bc_policy = ActionMaskedPolicy()
        bc_policy.load_state_dict(torch.load(bc_path, weights_only=True))
        print(f"Loaded BC policy from {bc_path}\n")
    else:
        print("=" * 60)
        print("Phase 1: Behavior Cloning")
        print("=" * 60)
        bc_policy = train_behavior_cloning(
            episodes=episodes,
            epochs=args.bc_epochs,
            batch_size=args.batch_size,
        )
        torch.save(bc_policy.state_dict(), bc_path)
        print(f"Saved BC policy to {bc_path}\n")

    # Phase 2: Actor-Critic CQL Fine-Tuning
    print("=" * 60)
    print("Phase 2: Actor-Critic CQL (SAC + Conservative Penalty)")
    print("=" * 60)
    agent = train_cql(
        episodes=episodes,
        bc_policy=bc_policy,
        epochs=args.cql_epochs,
        batch_size=args.batch_size,
    )
    cql_path = os.path.join(CHECKPOINTS_DIR, "cql_agent.pt")
    torch.save(agent.state_dict(), cql_path)
    print(f"Saved CQL agent to {cql_path}\n")

    print("Training complete!")
    print(f"  BC policy: {bc_path}")
    print(f"  CQL agent (actor-critic): {cql_path}")


if __name__ == "__main__":
    main()
