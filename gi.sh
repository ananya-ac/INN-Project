#!/bin/bash
# gi.sh - Run GI experiments with NLL, KLD, and JSD for two configurations
# Configuration 1: hidden_units=64, latent_dim=18, num_layers=3
# Configuration 2: hidden_units=403, latent_dim=2, num_layers=4

set -e  # Exit on error

# Common parameters
dataset='GI'
num_epochs=10
learning_rate=0.0003
network_type='glow'
prior_level=0
prior_type='none'
entropy_reg=0.01
seed=42
save_dir='gi_results'
fwd=0  # For train_fdiv.py (0=inverse problem)
prior='False'  # For train_fdiv.py

# Create save directory if it doesn't exist
mkdir -p "$save_dir"

# Configuration arrays
configs=(
    "64 18 3"   # Config 1: hidden_units=64, latent_dim=18, num_layers=3
    "403 2 4"   # Config 2: hidden_units=403, latent_dim=2, num_layers=4
)

echo "=========================================="
echo "  GI Experiments - NLL, KLD, JSD"
echo "=========================================="
echo ""
echo "Dataset: $dataset"
echo "Network type: $network_type"
echo "Seed: $seed"
echo "Save directory: $save_dir"
echo ""

# Run experiments for each configuration
for config in "${configs[@]}"; do
    read -r hidden_units latent_dim num_layers <<< "$config"

    echo "=========================================="
    echo "Configuration: hidden=$hidden_units, latent=$latent_dim, layers=$num_layers"
    echo "=========================================="

    # Run NLL training
    echo ""
    echo "[1/3] Training with NLL loss..."
    python train.py "NLL" "$dataset" "$num_layers" "$num_epochs" "$latent_dim" \
           "$seed" "$hidden_units" "$learning_rate" "$network_type" \
           "$prior_level" "$prior_type" "$save_dir" "$entropy_reg"

    # Run KLD training
    echo ""
    echo "[2/3] Training with KLD divergence..."
    python train_fdiv.py "KLD" "$prior" "$dataset" "$latent_dim" "$hidden_units" \
           "$seed" "$num_layers" "$fwd" "$save_dir" "$network_type"

    # Run JSD training
    echo ""
    echo "[3/3] Training with JSD divergence..."
    python train_fdiv.py "JSD" "$prior" "$dataset" "$latent_dim" "$hidden_units" \
           "$seed" "$num_layers" "$fwd" "$save_dir" "$network_type"

    echo ""
    echo "✓ Completed configuration: hidden=$hidden_units, latent=$latent_dim, layers=$num_layers"
    echo ""
done

echo "=========================================="
echo "  All experiments completed!"
echo "=========================================="
echo "Results saved to: $save_dir"
