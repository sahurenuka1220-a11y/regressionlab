"""RegressionLab: regression from scratch with NumPy."""
from .models import LinearModel, grid_search, kfold_scores, train_test_split
from .metrics import mae, r2_score, rmse

__all__ = ["LinearModel", "grid_search", "kfold_scores", "train_test_split",
           "mae", "r2_score", "rmse"]
__version__ = "1.0.0"
