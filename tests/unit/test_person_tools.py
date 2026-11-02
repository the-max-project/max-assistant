import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from max_assistant.tools.person_tools import PersonTools


@pytest.mark.asyncio
async def test_find_person_by_name_success(sample_user_info):
    """
    Verifies searching for a person node maps valid data arrays accurately
    when properties are properly flattened to mimic LangGraph tool injection.
    """
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {"person": {"id": "10", "firstName": "John", "lastName": "Doe"}}
            ]
        }
    )

    tools = PersonTools(db_client=db_client, llm=MagicMock())

    result = await tools.find_person_by_name( sample_user_info, first_name="John",)

    parsed = json.loads(result) if isinstance(result, str) else result
    assert parsed[0]["person"]["id"] == "10"
    assert parsed[0]["person"]["firstName"] == "John"


@pytest.mark.asyncio
async def test_find_person_by_name_requires_input(sample_user_info):
    """
    Verifies that passing empty string inputs returns the tool's
    designated fallback validation error contract.
    """
    tools = PersonTools(db_client=MagicMock(), llm=MagicMock())


    result = await tools.find_person_by_name(sample_user_info, "")

    parsed = json.loads(result) if isinstance(result, str) else result
    assert parsed["error"] == "Data parsing failed"


@pytest.mark.asyncio
async def test_find_person_by_title_success(sample_user_info):
    """Verifies finding a person by title works when payload arguments are flat."""
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {"person": {"id": "12", "firstName": "Bob", "lastName": "Smith"}}
            ]
        }
    )
    tools = PersonTools(db_client=db_client, llm=MagicMock())

    result = await tools.find_person_by_title("Engineer", sample_user_info)
    parsed = json.loads(result) if isinstance(result, str) else result
    assert parsed[0]["id"] == "12"


@pytest.mark.asyncio
async def test_get_user_info_internal_success():
    """Verifies capturing and mapping a user profile entry matching a designated username query."""
    db_client = MagicMock()
    fake_payload = {
        "user": {"id": 1, "firstName": "Max", "lastName": "Assistant"},
        "location": {"id": 2, "name": "Home"},
    }
    db_client.execute_query = AsyncMock(return_value={"data": [fake_payload]})

    tools = PersonTools(db_client=db_client, llm=MagicMock())

    # Act: This method natively returns a dict, bypassing json.loads requirement
    result = await tools.get_user_info_internal("robert_begg")

    assert result["user"]["firstName"] == "Max"
    assert result["location"]["name"] == "Home"


def test_get_tools_exposes_expected_names():
    """Validates the exact capability set made discoverable to the agent engine."""
    tools = PersonTools(db_client=MagicMock(), llm=MagicMock())
    tool_names = {tool.name for tool in tools.get_tools()}

    expected_tools = {
        "find_person_by_name",
        "find_person_by_title",
        "get_relationship_to_user",
        "query_relationship",
    }

    assert tool_names == expected_tools