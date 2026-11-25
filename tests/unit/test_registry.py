import logging
from unittest.mock import MagicMock

import pytest

from max_assistant.tools.base_provider import BaseToolProvider
from max_assistant.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry(db_client=MagicMock(), llm=MagicMock())


def test_register_provider_injects_dependencies_and_collects_tools(registry):
    tools = [object(), object()]

    class Provider(BaseToolProvider):
        def get_tools(self):
            return tools

    registry.register_provider(Provider)
    provider = registry.get_provider(Provider)
    assert provider.db_client is registry.db_client
    assert provider.llm is registry.llm
    assert registry.get_all_tools() == tools


def test_registration_collects_tools_in_order_and_lookup_returns_first_match(registry):
    class Provider(BaseToolProvider):
        def get_tools(self):
            return [self]

    class DerivedProvider(Provider):
        pass

    first = DerivedProvider(registry.db_client, registry.llm)
    second = Provider(registry.db_client, registry.llm)
    registry.register_instance(first)
    registry.register_instance(second)

    assert registry.get_all_tools() == [first, second]
    assert registry.get_provider(Provider) is first
    assert registry.get_provider(DerivedProvider) is first
    assert registry.get_provider(str) is None


def test_instance_without_get_tools_is_still_registered(registry):
    instance = object()
    registry.register_instance(instance)
    assert registry.get_provider(object) is instance
    assert registry.get_all_tools() == []


def test_non_provider_class_warns_but_registers(registry, caplog):
    tool = object()

    class OtherProvider:
        def __init__(self, db_client, llm):
            self.db_client, self.llm = db_client, llm

        def get_tools(self):
            return [tool]

    with caplog.at_level(logging.WARNING):
        registry.register_provider(OtherProvider)

    assert "does not inherit from BaseToolProvider" in caplog.text
    assert registry.get_provider(OtherProvider).db_client is registry.db_client
    assert registry.get_all_tools() == [tool]