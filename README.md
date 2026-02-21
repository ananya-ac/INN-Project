# VINA: Variational Invertible Neural Architectures

Code repository accompanying the paper **"VINA: Variational Invertible Neural Architectures"**.

## Setup

Create the conda environment and generate datasets:

```bash
./setup_env.sh
conda activate inn_final
```


### Geophysical Inversion Experiments

```bash
./gi.sh

```
Runs experiments with NLL, KLD, and JSD objectives for both model configurations. 



### Acknowledgment
The file robot_arm_2d.py is adapted from work by a-doering.
Original implementation available at:
https://github.com/a-doering/learning-inverse-kinematics.git


## Citation

```bibtex
@article{vina2025,
  title={VINA: Variational Invertible Neural Architectures},
  author={[Shubhanshu Shekhar, Mohammad Javad Khojasteh, Ananya Acharya, Tony Tohme, Kamal Youcef-Toumi]},
  year={2025}
}
```
