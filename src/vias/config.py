# config.py
import yaml
from dotmap import DotMap

_config = None


def load_config(yaml_file_path):
    global _config
    with open(yaml_file_path) as f:
        config_dict = yaml.safe_load(f)
    _config = DotMap(config_dict)
    return _config


def get_config():
    if _config is None:
        raise ValueError("Configuration not loaded. Call load_config() first.")
    return _config
