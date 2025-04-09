# INN Project

Use run.sh to run script to log RMSE between true end-effector position and the end-effector position obtained from sampling joint configurations from the INN.

Use run_fig.sh to save images for all combination of losses and priors. Usage: ./run.sh <prior_flag> <loss_function1> <loss_function2> ...

Run "pip install -r requirements.txt" in a conda/venv environment to re-create the environment.
Minimum python=3.11.10.
