# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines a dynamic, LLM-powered tool for answering general-purpose
questions against the Neo4j database.
"""
import json
import re
import logging
import asyncio
from typing import Annotated

from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState

from max_assistant.clients.neo4j_client import Neo4jClient, Neo4jCircuitBreakerError, Neo4jClientError
from max_assistant.agent.prompts import CYPHER_GENERATION_PROMPT
from max_assistant.tools.registry import BaseToolProvider
from max_assistant.utils.decorators import requires_db


logger = logging.getLogger(__name__)


class GeneralQueryTools(BaseToolProvider):
    """
    A toolset that uses an LLM to dynamically generate and execute
    Cypher queries for ad-hoc questions.
    """
    def __init__(self, db_client: Neo4jClient, llm: ChatOllama):
        """
        Initializes the toolset with a Neo4j client and an LLM.
        """
        super().__init__(db_client, llm)
        if llm is None:
            raise ValueError("GeneralQueryTools strictly requires an LLM instance.")

        # Create a raw text completion LLM pointing to the exact same model and host
        raw_llm = OllamaLLM(model=llm.model, base_url=llm.base_url)

        # Use a standard string template, avoiding Chat roles entirely
        RAW_CYPHER_PROMPT = PromptTemplate.from_template("""
        You are a Neo4j Cypher expert. Write a single, read-only Cypher query to answer the user's question.

        Schema Context:
        {schema}

        User Question: {question}

        CRITICAL RULES:
        - Output ONLY the raw Cypher query.
        - Wrap the query in a ```cypher code block.
        - DO NOT include <think> tags, reasoning, or explanations.
        """)

        # Bind the prompt to the raw completion model
        self.cypher_generation_chain = RAW_CYPHER_PROMPT | raw_llm
        logger.debug("GeneralQueryTools initialized with raw OllamaLLM generator.")

    @staticmethod
    def _parse_cypher_from_response(response_content: str) -> str:
        """
        Safely extracts a Cypher query from an LLM's markdown response.
        """
        # Look for a Cypher code block
        match = re.search(r"```(?:cypher|CYPHER)?\s*\n(.*?)\n\s*```", response_content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: if no code block, assume the whole response is the query
        # but clean it of common LLM "chatter"
        query = response_content.strip()
        if query.startswith("MATCH") or query.startswith("RETURN"):
            return query

        logger.warning(f"Could not parse Cypher from LLM response: {response_content}")
        # Return a query that will gracefully fail
        return "RETURN 'Error: Could not parse Cypher query from LLM response'"

    @requires_db
    async def answer_general_question(
            self,
            question: str,
            user_info: Annotated[dict, InjectedState("userinfo")]
    ) -> str:
        """
        Try to use this tool to answer ANY question about
        family members, support staff, relationships, locations, addresses, or personal history if no
        other specific tool applies. Use this for questions like "Does X have children?",
        "Where does Y live?", or "Who are my great-grandchildren?"
        """
        logger.info(f"Tool: answer_general_question for: {question}")

        # Convert the injected state dict into the string format your prompt expects
        #user_info_json = json.dumps(user_info)
        user_id = self._get_verified_user_id(user_info)
        params = {"user_id": user_id}

        try:
            # 1. Get the graph schema
            schema_str = await self.db_client.get_schema()

            # Check for error in schema fetching
            try:
                schema_data = json.loads(schema_str)
                if isinstance(schema_data, dict) and "error" in schema_data:
                    logger.error(f"Error retrieving graph schema: {schema_data}")
                    return json.dumps(
                        {"error": "Could not retrieve graph schema.", "details": schema_data.get("message")})
            except json.JSONDecodeError:
                logger.error(f"Failed to decode schema JSON: {schema_str}")
                return json.dumps({"error": "Failed to decode graph schema."})

            # 2. Generate the Cipher query
            logger.debug("Generating Cypher query...")
            try:
                # Force a 10-second hard limit on generation
                response_text = await asyncio.wait_for(
                    self.cypher_generation_chain.ainvoke({
                        "schema": schema_str,
                        "question": question,
                    }),
                    timeout=20.0
                )
            except asyncio.TimeoutError:
                logger.error("LLM Cypher generation timed out.")
                return json.dumps({"error": "Query generation took too long."})


            logger.info(f"RAW LLM CYPHER RESPONSE:\n{response_text}")

            cypher_query = self._parse_cypher_from_response(response_text)
            logger.info(f"Generated Cypher: {cypher_query}")

            # 3. Execute the query
            # We use params={} as the LLM is instructed to embed values
            result = await self.db_client.execute_query(cypher_query, params=params)

            # 4. Return the raw JSON string
            return json.dumps(result, indent=2, default=str)

        except Neo4jCircuitBreakerError as e:
            logger.warning(f"Circuit Breaker blocked general query: {e}")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The system database is currently offline. Do not attempt further queries. Inform the user you cannot access their data right now.",
                "details": str(e)
            })

        except Neo4jClientError as e:
            logger.error(f"Database error in dynamic query generation: {e}")
            return json.dumps({"error": "Database_Unavailable", "details": str(e)})

        except Exception as e:
            logger.error(f"Unexpected error in answer_general_question: {e}", exc_info=True)
            return json.dumps({"error": "Internal_Error", "details": str(e)})

    def get_tools(self) -> list:
        """
        Returns a list of all tool methods bound to this instance.
        """
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.answer_general_question,
                name="answer_general_question",
                description=self.answer_general_question.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
        ]