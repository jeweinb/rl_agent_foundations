#!/usr/bin/env python3
"""CLI script to train dynamics and reward world models."""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train_dynamics import train_dynamics_model
from models.train_reward import train_reward_model
from config import CHECKPOINTS_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train world models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Training Dynamics Model")
    print("=" * 60)
    dynamics = train_dynamics_model(epochs=args.epochs, batch_size=args.batch_size)
    dynamics_path = os.path.join(CHECKPOINTS_DIR, "dynamics_model.pt")
    torch.save(dynamics.state_dict(), dynamics_path)
    print(f"Saved to {dynamics_path}\n")

    print("=" * 60)
    print("Training Reward Model")
    print("=" * 60)
    reward = train_reward_model(epochs=args.epochs, batch_size=args.batch_size)
    reward_path = os.path.join(CHECKPOINTS_DIR, "reward_model.pt")
    torch.save(reward.state_dict(), reward_path)
    print(f"Saved to {reward_path}\n")

    print("Both world models trained and saved.")


if __name__ == "__main__":
    main()
