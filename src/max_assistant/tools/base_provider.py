# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""

"""
import json
import logging
from typing import List, Type, Dict, Any

from pydantic import BaseModel, ValidationError
from langchain_core.tools import BaseTool, ToolException
from langchain_ollama import ChatOllama

from max_assistant.clients.neo4j_client import Neo4jClient, Neo4jClientError, Neo4jCircuitBreakerError
from max_assistant.utils.decorators import requires_db

logger = logging.getLogger(__name__)


class BaseToolProvider:
    """
    Abstract base class for a class that provides tools.
    This helps with type hinting and structure.
    """
    def __init__(self, db_client: Neo4jClient, llm: ChatOllama | None = None):
        self.db_client = db_client
        self.llm = llm

    @staticmethod
    def _get_user_id(user_info: Dict[str, Any]) -> str | None:
        """Centralized helper to safely retrieve user IDs from the session context."""
        if not user_info:
            return None
        return user_info.get("user", {}).get("id")

    def _get_verified_user_id(self, user_info: Dict[str, Any]) -> str:
        """
        Extracts the user ID. If missing or unauthenticated, immediately raises
        a ToolException, aborting tool execution and notifying LangGraph.
        """
        user_id = self._get_user_id(user_info)
        if not user_id:
            raise ToolException(
                "System error. User ID is missing."
            )
        return user_id

    @staticmethod
    def format_system_tool_error(error: ToolException) -> str:
        """
        Global LangGraph tool error handler. Wraps raised ToolExceptions
        into a consistent structured JSON contract for the LLM.
        """
        error_text = str(error)

        logger.warning(f"Tool execution intercepted by global handler: {error_text}")
        return json.dumps({
            "success": False,
            "error": "Tool_Execution_Failed",
            "message": error_text,
            "instruction": (
                "Analyze the message above. Communicate the failure to the user gracefully. "
                "Do not attempt to retry this tool with the exact same parameters unless the "
                "user corrects them or provides missing authentication."
            )
        }, indent=2)

    def get_tools(self) -> List[BaseTool]:
        raise NotImplementedError

    @requires_db
    async def _query_and_validate_nodes(
            self,
            query: str,
            model_class: Type[BaseModel],
            result_key: str,
            params: dict | None = None,
    ) -> str:
        """
        Executes a query, validates results against a Pydantic model,
        and returns a JSON string gracefully handling all DB/Validation errors.
        """
        logger.debug(f"Executing query for model: {model_class.__name__}")

        try:
            result = await self.db_client.execute_query(query, params or {})

            data_list = result.get("data", [])
            raw_nodes = [item[result_key] for item in data_list if isinstance(item, dict) and result_key in item]

            if len(raw_nodes) != len(data_list):
                logger.warning(f"Some records were skipped because they missed the key '{result_key}'")

            validated_nodes = [model_class.model_validate(node) for node in raw_nodes]
            if not validated_nodes:
                logger.info(f"No valid records found")
                return "No valid records found."

            logger.info(f"Successfully validated {len(validated_nodes)} records")

            return json.dumps(
                [node.model_dump(mode='json') for node in validated_nodes],
                indent=2,
            )

        except Neo4jCircuitBreakerError as e:
            # 1. Catch the Fast-Fail FIRST
            logger.warning(f"Circuit Breaker blocked {model_class.__name__} query: {e}")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The system database is currently offline. Do not attempt further queries. Inform the user you cannot access their data right now.",
                "details": str(e)
            })

        except Neo4jClientError as e:
            # 2. Catch standard query/driver errors
            logger.error(f"Database error in {model_class.__name__} query: {e}")
            return json.dumps({"error": "Database_Unavailable", "details": str(e)})

        except ValidationError as e:
            logger.error(f"Validation error for {model_class.__name__}: {e.errors()[:3]}")
            return json.dumps({
                "error": "Data_Validation_Failed",
                "details": "The data returned from the database did not match the expected system schema."
            })

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return json.dumps({"error": "Data parsing failed", "details": str(e)})

    @requires_db
    async def _safe_execute_query(self, query: str, params: dict | None = None) -> str:
        """
        Executes a raw query and safely returns the JSON stringified result.
        Ideal for writing operations or queries that don't need Pydantic validation.
        """
        try:
            result = await self.db_client.execute_query(query, params or {})
            return json.dumps(result, indent=2, default=str)

        except Neo4jCircuitBreakerError as e:
            logger.warning(f"Circuit Breaker blocked raw query: {e}")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The system database is currently offline. Do not attempt further queries. Inform the user you cannot access their data right now.",
                "details": str(e)
            })

        except Neo4jClientError as e:
            logger.error(f"Database error during raw query execution: {e}")
            return json.dumps({"error": "Database_Unavailable", "details": str(e)})

        except Exception as e:
            logger.error(f"Unexpected error during raw query execution: {e}", exc_info=True)
            return json.dumps({"error": "Internal_Error", "details": str(e)})