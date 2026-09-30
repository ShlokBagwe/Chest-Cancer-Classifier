from pathlib import Path
import os
import yaml
from ChestCancer import logger
from box.exceptions import BoxValueError
from box import ConfigBox
from ensure import ensure_annotations

"""
ConfigBox allows us to access dictionary values using dot notation instead of nested [].
config["artifacts"]["data_ingestion"]["root_dir"] → config.artifacts.data_ingestion.root_dir 
"""


def read_yaml(yaml_path: Path) -> ConfigBox:
    """
    reads yaml files and returns 
    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    
    """

    try:
        with open(yaml_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {[yaml_path]}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    
    """ 
    Create List of Directories
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info("created directory at: {path}")

    