from dataclasses import dataclass
from pathlib import Path

from util import get_root

@dataclass
class ParamDir():
    ROOT = get_root()
    DATA_DIR =  ROOT / "data/winequality-red.csv"
    OUTPUT_DIR = ROOT / "output/"

@dataclass
class ParamTrain():
    train_ratio = .8


@dataclass
class ParamEval():
    metric_name = "mean_square_error"