"""Configuration management module."""

import os
from typing import Dict, Any, List
import yaml


class Config:
    """Configuration class for accessing YAML config values."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to config values."""
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        return self._config


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config object with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['tei', 'chromadb', 'processing', 'data', 'logging']
    missing_sections = [s for s in required_sections if s not in config_dict]
    if missing_sections:
        raise ValueError(f"Missing required config sections: {missing_sections}")

    # Validate TEI config
    if 'api_url' not in config_dict['tei']:
        raise ValueError("TEI config missing 'api_url'")

    # Validate data files config
    if 'files' not in config_dict['data'] or not config_dict['data']['files']:
        raise ValueError("Data config missing 'files' list")

    for file_config in config_dict['data']['files']:
        if 'name' not in file_config or 'collection' not in file_config:
            raise ValueError("Each file config must have 'name' and 'collection'")

    return Config(config_dict)


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration values.

    Args:
        config: Config object to validate

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []

    # Check TEI configuration
    if config.tei.batch_size < 1:
        warnings.append("TEI batch_size should be at least 1")
    if config.tei.batch_size > 256:
        warnings.append("TEI batch_size > 256 may cause performance issues")
    if config.tei.timeout < 10:
        warnings.append("TEI timeout < 10 seconds may be too short")

    # Check processing configuration
    if config.processing.csv_chunk_size < 100:
        warnings.append("csv_chunk_size < 100 may cause excessive I/O")
    if config.processing.chroma_batch_size < 10:
        warnings.append("chroma_batch_size < 10 may cause slow inserts")

    # Check data paths
    input_dir = config.data.input_dir
    if not os.path.exists(input_dir):
        warnings.append(f"Data input directory does not exist: {input_dir}")
    else:
        for file_config in config.data.files:
            file_path = os.path.join(input_dir, file_config['name'])
            if not os.path.exists(file_path):
                warnings.append(f"Data file not found: {file_path}")

    # Check directories exist or can be created
    log_dir = config.logging.log_dir
    if not os.path.exists(log_dir):
        warnings.append(f"Log directory will be created: {log_dir}")

    persist_dir = config.chromadb.persist_directory
    if not os.path.exists(persist_dir):
        warnings.append(f"ChromaDB directory will be created: {persist_dir}")

    return warnings
