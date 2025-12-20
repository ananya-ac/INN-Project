#!/usr/bin/env python
"""
Generate training data for inverse kinematics.
This creates the dataset file expected by loader.py
"""

import argparse
from robot_arm_2d import RobotArm2d


def main():
    parser = argparse.ArgumentParser(description='Generate inverse kinematics training data')
    parser.add_argument('--num-forward', type=int, default=100,
                        help='Number of forward configurations (default: 100)')
    parser.add_argument('--num-inverse', type=int, default=1000,
                        help='Number of inverse samples per configuration (default: 1000)')
    parser.add_argument('--lengths', type=float, nargs='+', default=[0.5, 0.5, 1.0],
                        help='Arm segment lengths (default: [0.5, 0.5, 1.0])')
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.25, 0.5, 0.5, 0.5],
                        help='Joint parameter standard deviations (default: [0.25, 0.5, 0.5, 0.5])')

    args = parser.parse_args()

    print("=" * 50)
    print("  Inverse Kinematics Data Generation")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Arm lengths: {args.lengths}")
    print(f"  Sigmas: {args.sigmas}")
    print(f"  Forward configurations: {args.num_forward}")
    print(f"  Inverse samples per config: {args.num_inverse}")
    print(f"  Total samples: {args.num_forward * args.num_inverse:,}")
    print(f"\nExpected output file:")
    print(f"  ./data/inverse_data_{len(args.sigmas)}_{args.num_forward}_{args.num_inverse}.pickle")
    print()

    # Create robot arm
    arm = RobotArm2d(lengths=args.lengths, sigmas=args.sigmas)

    # Generate and save data
    print("Starting data generation...")
    arm.generate_data(arm.sample_priors(args.num_forward), args.num_inverse)

    print("\n✓ Data generation complete!")
    print("\nYou can now run training with:")
    print("  python train.py NLL ik_robotics 8 100 2 42 256 1e-4 glow 1.0 normal results 0.01")


if __name__ == "__main__":
    main()
