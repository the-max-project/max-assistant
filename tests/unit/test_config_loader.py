import os
import tempfile
import yaml
from pathlib import Path
from max_assistant.config_loader import load_ollama_config, get_ollama_params, get_model_name_from_config


def test_load_ollama_config_no_file():
    """Test loading config when no file exists"""
    # Create a temporary directory and change to it
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            result = load_ollama_config()
            assert isinstance(result, dict)
            assert result == {}
        finally:
            os.chdir(old_cwd)


def test_load_ollama_config_with_file():
    """Test loading config from a valid YAML file"""
    # Create a temporary directory and change to it
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            # Create a test config file
            config_data = {
                'default': {
                    'temperature': 0.7,
                    'top_p': 0.9
                },
                'models': {
                    'test-model': {
                        'temperature': 0.5
                    }
                },
                'default_model': 'test-model'
            }
            
            with open('ollama_config.yaml', 'w') as f:
                yaml.dump(config_data, f)
            
            result = load_ollama_config()
            assert isinstance(result, dict)
            assert 'default' in result
            assert 'models' in result
            assert result['default'] == {'temperature': 0.7, 'top_p': 0.9}
            assert result['models']['test-model'] == {'temperature': 0.5}
            # Note: default_model is not included in the returned dict from load_ollama_config
            # but can be accessed through get_model_name_from_config function
        finally:
            os.chdir(old_cwd)


def test_get_ollama_params():
    """Test getting Ollama parameters for a model"""
    config = {
        'default': {
            'temperature': 0.7,
            'top_p': 0.9
        },
        'models': {
            'test-model': {
                'temperature': 0.5,
                'host': 'localhost:11434'
            }
        }
    }
    
    # Test with a specific model
    params = get_ollama_params('test-model', config)
    assert params['temperature'] == 0.5  # Model-specific override
    assert params['top_p'] == 0.9  # Default value
    assert params['host'] == 'localhost:11434'  # Model-specific value
    
    # Test with default model (no specific settings)
    params = get_ollama_params('non-existent-model', config)
    assert params['temperature'] == 0.7
    assert params['top_p'] == 0.9


def test_get_model_name_from_config():
    """Test getting model name from config and environment"""
    # Test with empty config and no env var
    result = get_model_name_from_config({}, '')
    assert result == 'llama3'
    
    # Test with default_model in config
    config = {'default_model': 'gemma4'}
    result = get_model_name_from_config(config, '')
    assert result == 'gemma4'
    
    # Test with environment var (should override config)
    result = get_model_name_from_config(config, 'gemma2')
    assert result == 'gemma2'


def test_load_ollama_config_invalid_yaml():
    """Test loading config with invalid YAML content"""
    # Create a temporary directory and change to it
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            # Create an invalid YAML file
            with open('ollama_config.yaml', 'w') as f:
                f.write('invalid: yaml: content:')
            
            result = load_ollama_config()
            assert isinstance(result, dict)
            assert result == {}  # Should return empty dict on error
        finally:
            os.chdir(old_cwd)