from dataclasses import dataclass, field
import torch

# Device setup
var = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
y_noise_scale = 0.1
zeros_noise_scale = 5e-2
prior_scale = 1
num_samples = 1000


# =====================
# Dataset Configurations
# =====================



@dataclass
class IKConfig:
    ndim_x: int = 4
    ndim_y: int = 2
    num_joints: int = 4
    lengths: list = field(default_factory=lambda: [0.5, 0.5, 1])
    sigmas: list = field(default_factory=lambda: [0.25, 0.5, 0.5, 0.5])


@dataclass
class GIConfig:
    ndim_x: int = 72
    ndim_y: int = 70
    param_dim: int = 2





# =====================
# Dataloader Config
# =====================
@dataclass
class DataLoaderConfig:
    batch_size: int = 512
    

# =====================
# Training Config
# =====================
@dataclass
class TrainingConfig:
    n_epochs: int = 10
    val_split: float = 0.8


# =====================
# Optimizer Config
# =====================
@dataclass
class OptimizerConfig:
    lr: float = 1e-6
    l2_reg: float = 2e-5
    betas: tuple = (0.8, 0.9)
    eps: float = 1e-6


# =====================
# Loss Weights Config
# =====================
@dataclass
class LossWeightsConfig:
    """Configuration for loss function weights to eliminate magic numbers."""
    # NLL loss weights
    nll_fit_l2_weight: float = 25.0
    nll_reconstruction_weight: float = 5.0

    # Wasserstein loss weights
    wasserstein_fit_l2_weight: float = 10.0
    wasserstein_reconstruction_weight: float = 10.0
    wasserstein_distance_weight: float = 100.0

    # Prior loss weights (dataset-specific)
    prior_ik_robotics_weight: float = 1.0

    # F-divergence training weights
    fdiv_reconstruction_weight: float = 0.01


# =====================
# Paths Config
# =====================
@dataclass
class PathsConfig:
    """Configuration for file paths to eliminate hardcoded strings."""
    # Data directories
    data_dir: str = "./data"
    gi_data_dir: str = "GI_Data"

    # Dataset files
    inverse_data_file: str = "./data/inverse_data_4_500_1000.pickle"
    gi_data_file: str = "GI_Data/data2D.mat"

    # GI fixed point tensors
    x_star_file: str = "x_star_tensor.pt"
    y_star_file: str = "y_star_tensor.pt"
    x_star_legacy: str = "x_star.pt"

    # Output/logging
    gi_mse_nll_log: str = "gi_mse_nll.log"
    training_time_log: str = "training_time_nll.log"
    observation_6_log: str = "observation_6_moments.log"




