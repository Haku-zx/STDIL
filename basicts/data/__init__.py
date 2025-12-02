import os

# from easytorch.utils.registry import scan_modules
from .registry import scan_modules

from .registry import SCALER_REGISTRY
from .dataset import TimeSeriesForecastingDataset

__all__ = ["SCALER_REGISTRY", "TimeSeriesForecastingDataset"]
# print("Current working directory:", os.getcwd())
# print("Current file path:", __file__)
# Current working directory: E:\AA_D盘\AA未复现代码\交通流预测未复现代码\这个\STD-MAE-main\STD-MAE-main\stdmae
# Current file path: E:\AA_D盘\AA未复现代码\交通流预测未复现代码\这个\STD-MAE-main\STD-MAE-main\basicts\data\__init__.py
scan_modules(os.getcwd(), __file__, ["__init__.py", "registry.py"])

