#!/bin/bash
losses=('wasserstein')  
divergence='KLD'
num_layers=(4)
num_epochs=(10)
latent_dims=(2 4 5 12 18)
hidden_units=(128)
learning_rate=0.0003
network_type='glow'
prior_levels=(0)
seeds=({40..60})
save_dir='latent_observation'
prior_type=('none')
entropy_reg=0.01
for loss in "${losses[@]}"; do
    for latent_dim in "${latent_dims[@]}"; do
        for seed in "${seeds[@]}"; do
            for num_epoch in "${num_epochs[@]}"; do
                for hidden_unit in "${hidden_units[@]}"; do
                    for num_layer in "${num_layers[@]}"; do
                        for prior_level in "${prior_levels[@]}"; do
                                echo "Running train with loss: $loss, latent_dim: $latent_dim, hidden_units: $hidden_unit, num_layers: $num_layer, seed: $seed, prior_level: $prior_level"
                                python train.py "$loss" "ik_robotics" "$num_layer" "$num_epoch" "$latent_dim" "$seed" "$hidden_unit" "$learning_rate" "$network_type" "$prior_level" "none" "$save_dir" "$entropy_reg"
                                python train_fdiv.py "$divergence" "$prior" "ik_robotics" "$latent_dim" "$hidden_unit" "$seed" "$num_layer" "0" "$save_dir" "$network_type"

                        done
                    done
                done
            done
        done
    done
done
                                
#latent_dim=2, hidden_units=403, num_layers=4