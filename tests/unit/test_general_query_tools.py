import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.tools.general_query_tools import GeneralQueryTools


def test_parse_cypher_from_markdown_block():
    """Verifies parsing logic handles standard markdown fenced code blocks cleanly."""
    tools = GeneralQueryTools(db_client=MagicMock(), llm=MagicMock())
    response = "```cypher\nMATCH (n) RETURN n\n```"
    assert tools._parse_cypher_from_response(response) == "MATCH (n) RETURN n"


def test_parse_cypher_from_plain_query():
    """Verifies parsing logic leaves standard queries untouched if no markdown exists."""
    tools = GeneralQueryTools(db_client=MagicMock(), llm=MagicMock())
    response = "MATCH (n) RETURN n LIMIT 1"
    assert tools._parse_cypher_from_response(response) == "MATCH (n) RETURN n LIMIT 1"


def test_parse_cypher_from_invalid_response_returns_safe_error_query():
    """Verifies that unstructured conversation triggers a fallback string error statement."""
    tools = GeneralQueryTools(db_client=MagicMock(), llm=MagicMock())
    response = "This is not a query"
    parsed = tools._parse_cypher_from_response(response)
    assert parsed == "RETURN 'Error: Could not parse Cypher query from LLM response'"


@pytest.mark.asyncio
async def test_answer_general_question_happy_path():
    """
    Verifies that a valid question successfully triggers Cypher generation
    and executes it directly against the Neo4j client cluster.
    """
    db_client = MagicMock()
    db_client.get_schema = AsyncMock(return_value=json.dumps({"node_labels": ["User"]}))
    db_client.execute_query = AsyncMock(return_value={"data": [{"name": "A"}]})

    # Mock the internal prompt/LLM pipeline execution
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="```cypher\nMATCH (n) RETURN n\n```"))

    tools = GeneralQueryTools(db_client=db_client, llm=MagicMock())

    # Patch the bound runtime chain object and pass a real Python dict for context
    with patch.object(tools, "cypher_generation_chain", mock_chain):
        result = await tools.answer_general_question(
            question="who is my father?",
            user_info={"user": {"id": 1}}  # FIX: Native dictionary type
        )

    parsed = json.loads(result)
    assert parsed == {"data": [{"name": "A"}]}
    #TODO: the schema is no longer being used, awaiting further refactoring of the general query tool
    #db_client.get_schema.assert_awaited_once()
    db_client.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_answer_general_question_database_unavailable_error():
    """Verifies that a Neo4jClientError is captured and converted to a safe JSON string payload."""
    from max_assistant.clients.neo4j_client import Neo4jClientError

    db_client = MagicMock()
    db_client.get_schema = AsyncMock(return_value=json.dumps({"node_labels": ["User"]}))
    # Simulate a standard database execution fault scenario
    db_client.execute_query = AsyncMock(side_effect=Neo4jClientError("Connection refused by host"))

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="MATCH (n) RETURN n"))

    tools = GeneralQueryTools(db_client=db_client, llm=MagicMock())

    with patch.object(tools, "cypher_generation_chain", mock_chain):
        result = await tools.answer_general_question(
            question="show all entries",
            user_info={"user": {"id": 2}}  # FIX: Native dictionary type
        )

    parsed = json.loads(result)
    assert parsed["error"] == "Database_Unavailable"
    assert "Connection refused" in parsed["details"]


def test_get_tools_exposes_expected_names():
    """Enforces that GeneralQueryTools registers the exact public capability names list."""
    tools = GeneralQueryTools(db_client=MagicMock(), llm=MagicMock())
    tool_names = {tool.name for tool in tools.get_tools()}

    expected_tools = {
        "answer_general_question"
    }

    assert tool_names == expected_tools