#!/bin/bash

divergences=("KLD" "JSD" "RKL")
bool_values=(False)
seeds=({40..60})
latent_dim=(14)
hidden_units=(128 64)
num_layers=(4 6)
save_dir="observation_relative_performance"
network_type="glow"
for div in "${divergences[@]}"; do
    for prior in "${bool_values[@]}"; do
        for lat_dim in "${latent_dim[@]}"; do
            for hidden in "${hidden_units[@]}"; do
                for layers in "${num_layers[@]}"; do
                    for seed in "${seeds[@]}"; do
                        echo "Running train_fdiv with divergence: $div, prior: $prior, latent_dim: $lat_dim, hidden_units: $hidden, num_layers: $layers, seed: $seed"
                        python train_fdiv.py "$div" "$prior" "ik_robotics" "$lat_dim" "$hidden" "$seed" "$layers" "0" "$save_dir" "$network_type"
                    done
                done
            done
        done
    done
done

#latent_dim=18, hidden_units=64, num_layers=3 #best for water depth
#latent_dim=4, hidden_units=427, num_layers=4 #best for speed