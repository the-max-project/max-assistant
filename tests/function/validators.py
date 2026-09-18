# tests/function/validators.py

import re
import json
from typing import List
from ollama import AsyncClient
from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.config import OLLAMA_MODEL_NAME


# Keep synchronous validators exactly as they are
def assert_substring_present(expected_substring: str):
    def validator(response_text: str, db_client: Neo4jClient):
        assert expected_substring.lower() in response_text.lower(), \
            f"Expected substring '{expected_substring}' missing from response: '{response_text}'"

    return validator


def assert_substrings_present(expected_substrings: List[str], match_all: bool = True):
    def validator(response_text: str, db_client: Neo4jClient):
        normalized_response = response_text.lower()
        missing = [sub for sub in expected_substrings if sub.lower() not in normalized_response]
        if match_all and missing:
            assert False, f"Substring validation failed! Response was missing required tokens: {missing}."
        elif not match_all and len(missing) == len(expected_substrings):
            assert False, f"Any-match validation failed! Response did not match any of: {expected_substrings}."

    return validator


# --- MAKE THESE TWO FUNCTIONS NATIVELY ASYNCHRONOUS ---

def assert_neo4j_node_exists(label: str, property_key: str, property_value: str):
    """Queries Neo4j asynchronously using the existing test runner event loop."""

    async def validator(response_text: str, db_client: Neo4jClient):
        query = f"MATCH (n:{label} {{{property_key}: $value}}) RETURN count(n) AS node_count"
        records, _, _ = await db_client.execute_query(query, {"value": property_value})
        count = records[0]["node_count"] if records else 0
        assert count > 0, f"Graph verification failed! No node found with label ':{label}' where {property_key}='{property_value}'"

    return validator


def assert_semantic_criteria(criteria: List[str]):
    """Leverages a small local LLM natively awaiting Ollama responses."""

    async def validator(response_text: str, db_client: Neo4jClient, model_name: str = None):
        system_prompt = (
            "You are a strict QA Automated Test Engine. Your assignment is to evaluate a raw language response "
            "against an engineering criteria checklist. Analyze the assertions meticulously.\n\n"
            "You must output ONLY a valid, raw JSON object matching this schema exactly:\n"
            '{"passed": boolean, "reason": "Clear context description explaining the pass/fail determination"}'
        )

        formatted_criteria = "\n".join(f"- {rule}" for rule in criteria)
        user_prompt = f"""-- CRITERIA TO ENFORCE --
{formatted_criteria}

-- ACTUAL RESPONSE TO EVALUATE --
"{response_text}"
"""

        # Intercepts host from OLLAMA_HOST environment layer automatically
        client = AsyncClient()
        active_model = model_name or OLLAMA_MODEL_NAME

        response = await client.chat(
            active_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            format="json",  # Enforce native JSON mode
            keep_alive=-1,
            options={
                "temperature": 0.0,
                "num_predict": 1024,  # Evaluator response is short; prevents runaway generations
            }
        )
        raw_output = response["message"]["content"].strip()

        try:
            clean_json = re.sub(r"^```json|```$", "", raw_output, flags=re.IGNORECASE).strip()
            result = json.loads(clean_json)
        except json.JSONDecodeError:
            assert False, f"Semantic Judge panicked! Failed to parse structured JSON from model evaluation output: '{raw_output}'"

        assert result.get("passed") is True, (
            f"Semantic Validation Failure on Model Under Test!\n"
            f"Reasoning: {result.get('reason')}\n"
            f"Evaluated Response: '{response_text}'"
        )

    return validator


# Add this code to the bottom of tests/function/validators.py

# A translation mapping to bridge flat YAML declarations directly to Python logic
VALIDATOR_REGISTRY = {
    "substring_present": assert_substring_present,
    "substrings_present_all": lambda substrings: assert_substrings_present(substrings, match_all=True),
    "substrings_present_any": lambda substrings: assert_substrings_present(substrings, match_all=False),
    "semantic_criteria": assert_semantic_criteria,
}


def build_steps_from_yaml(yaml_steps: list) -> list:
    """
    Transforms streamlined key-value YAML step definitions into
    the executable dictionary structure required by execute_scenario_workflow.
    """
    processed_steps = []

    for step in yaml_steps:
        validators = []
        for validator_wrapper in step.get("validators", []):
            # Extract key and configuration (e.g. "substrings_present_all" and ["Emily", "Ryan"])
            validator_name, config = next(iter(validator_wrapper.items()))

            if validator_name not in VALIDATOR_REGISTRY:
                raise ValueError(f"Unknown validator wrapper key in YAML: '{validator_name}'")

            factory_function = VALIDATOR_REGISTRY[validator_name]

            # Direct parsing pipeline since everything is a flat list or primitive string
            if isinstance(config, list):
                # Special handling for semantic criteria which expects a list parameter directly
                if validator_name == "semantic_criteria":
                    validator_instance = factory_function(config)
                else:
                    validator_instance = factory_function(*config) if len(config) == 1 else factory_function(config)
            else:
                validator_instance = factory_function(config)

            validators.append(validator_instance)

        processed_steps.append({
            "user_input": step["user_input"],
            "validators": validators
        })

    return processed_steps