# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
YAML configuration loader for Ollama parameters in Max-Assistant.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def load_ollama_config() -> Dict[str, Any]:
    """
    Load Ollama configuration from YAML file with fallbacks.
    
    Returns:
        Dict containing configuration for ChatOllama initialization
    """
    # Look for config file in the current working directory (where service runs)
    config_path = Path("ollama_config.yaml")
    
    # If no config file exists, return empty dict to use defaults
    if not config_path.exists():
        logger.info("No ollama_config.yaml found, using defaults")
        return {}
        
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) or {}
        
        # Validate structure and return
        return {
            'default': config.get('default', {}),
            'models': config.get('models', {})
        }
    except Exception as e:
        logger.warning(f"Failed to load ollama_config.yaml: {e}")
        return {}

def get_ollama_params(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get merged configuration for a specific model.
    
    Args:
        model_name: Name of the model to get parameters for
        config: Loaded configuration dictionary
        
    Returns:
        Merged parameters dict with default values and model-specific overrides
    """
    # Get default settings
    default_settings = config.get('default', {})
    
    # Get model-specific settings, if any
    model_settings = config.get('models', {}).get(model_name, {})
    
    # Merge with priority: model > default > hardcoded (existing logic)
    merged_params = default_settings.copy()
    merged_params.update(model_settings)
    
    # Handle the options section separately for flexible parameter passing
    options = merged_params.pop('options', {})
    
    # Construct final params dict for ChatOllama
    final_params = {}
    
    # Add regular parameters that are direct ChatOllama args
    for key, value in merged_params.items():
        if key != 'options':  # Skip the options key itself
            final_params[key] = value
    
    # Add options as a separate dict that will be passed to the constructor
    final_params['options'] = options
    
    return final_params

def get_model_name_from_config(config: Dict[str, Any], env_model_name: str) -> str:
    """
    Determine which model name to use based on environment variable and config.
    
    Args:
        config: Loaded configuration dictionary
        env_model_name: Model name from OLLAMA_MODEL_NAME environment variable
        
    Returns:
        Final model name string
    """
    # If environment variable is set, use it (overrides config)
    if env_model_name:
        return env_model_name
    
    # Otherwise, get model from default_model or fall back to "llama3"
    default_model = config.get('default_model', 'llama3')
    return default_model