import sys
import os.path as P
sys.path.insert(0, P.abspath(P.join(P.dirname(__file__), '../lib/')))
sys.path.insert(0, P.abspath(P.join(P.dirname(__file__), '../experimentations/')))

from src.config import default_config, parse_config
from src.datasets import load_dataset
from run_train import setup_model
