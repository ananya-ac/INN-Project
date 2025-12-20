

#!/bin/bash
seeds=({40..50})
alpha_values=(1 2 3 4 5 6 7 8 9 10)
for seed in "${seeds[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        echo "Running pareto_exp with seed: $seed, alpha: $alpha"
        python pareto_exp.py "$seed" "$alpha"
    done
done