# max-assistant

The core "brain" for the assistant using LangGraph

## YAML Configuration

Starting from version 1.0, Max-Assistant supports external YAML configuration files to customize Ollama model parameters.

### Configuration File Location

Create a file named `ollama_config.yaml` in the root directory of the max-assistant service (where you run the application):

```
services/max-assistant/ollama_config.yaml
```

This is typically the working directory when running the service, which can be:
- The local directory where `main.py` resides
- The `services/max-assistant/` subdirectory of the project root

The configuration file supports three sections:

### Configuration Format

```yaml
# Default model name to use when OLLAMA_MODEL_NAME is not set
default_model: "gemma4:26B"

# Default settings applied to all models without specific configuration
default:
  temperature: 0.0
  timeout: 120.0
  keep_alive: "-1"
  thinking_level: "balanced"
  # Additional ChatOllama parameters can be specified in the 'options' section  
  options:
    num_ctx: 8192

# Model-specific configurations
models:
  llama3:
    temperature: 0.7
    timeout: 180.0
    thinking_level: "detailed"
    options:
      num_ctx: 16384
      numa: true
  mistral:
    temperature: 0.3
    thinking_level: "concise"
    options:
      num_ctx: 4096
      top_k: 50
```

### Behavior

1. The `OLLAMA_MODEL_NAME` environment variable takes highest priority and overrides any configuration
2. If no environment variable is set, the `default_model` value is used
3. Model-specific settings override defaults when a matching model name is found
4. All parameters in the `options` section are passed directly to the ChatOllama constructor
5. Any new ChatOllama parameter can be added to options without code changes

### Usage

The configuration file is automatically loaded during application startup and respects existing environment variable precedence.
