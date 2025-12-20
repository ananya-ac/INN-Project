#!/bin/bash
losses=('NLL')  
num_layers=(4)
num_epochs=(10)
latent_dims=(14)
hidden_units=(128)
learning_rate=0.0003
network_type='glow'
prior_levels=(0 1 100)
seeds=(42)
save_dir='prior_observation'
prior_type=('normal' 'uniform')
for loss in "${losses[@]}"; do
    for latent_dim in "${latent_dims[@]}"; do
        for seed in "${seeds[@]}"; do
            for num_epoch in "${num_epochs[@]}"; do
                for hidden_unit in "${hidden_units[@]}"; do
                    for num_layer in "${num_layers[@]}"; do
                        for prior_level in "${prior_levels[@]}"; do
                            for p_type in "${prior_type[@]}"; do
                                echo "Running train with loss: $loss, latent_dim: $latent_dim, hidden_units: $hidden_unit, num_layers: $num_layer, seed: $seed, prior_level: $prior_level"
                                python train.py "$loss" "ik_robotics" "$num_layer" "$num_epoch" "$latent_dim" "$seed" "$hidden_unit" "$learning_rate" "$network_type" "$prior_level" "$p_type" "$save_dir" "0.0"
                            done
                        done
                    done
                done
            done
        done
    done
done
                                
#latent_dim=2, hidden_units=403, num_layers=4