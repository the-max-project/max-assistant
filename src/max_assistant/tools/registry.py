# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines a central registry for managing tool providers, enabling
tool discovery, initialization, and collection.

The `ToolRegistry` class orchestrates the overall lifecycle of tool providers,
including registration, dependency injection into providers' constructors, and
access to all tools provided by the registered providers.
"""
import logging
from typing import List, Optional, Type, TypeVar

from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.tools.base_provider import BaseToolProvider

T = TypeVar('T')

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    A registry to manage the collection and initialization of tool providers.
    """

    def __init__(self, db_client: Neo4jClient, llm: ChatOllama):
        self.db_client = db_client
        self.llm = llm
        self._providers: List[BaseToolProvider] = []
        self._tools: List[BaseTool] = []

    def register_provider(self, provider_class: Type[BaseToolProvider]):
        """
        Initializes and registers a tool provider.
        The provider class is instantiated with the db_client and llm.
        """
        if not issubclass(provider_class, BaseToolProvider):
            logger.warning(
                f"Class {provider_class.__name__} does not inherit from BaseToolProvider. "
                "Registration might not work as expected."
            )

        # Instantiate the provider, passing the necessary clients.
        provider_instance = provider_class(db_client=self.db_client, llm=self.llm)
        self._providers.append(provider_instance)
        # Eagerly collect tools upon registration.
        new_tools = provider_instance.get_tools()
        self._tools.extend(new_tools)
        logger.debug(f"Registered {len(new_tools)} tools from {provider_class.__name__}.")

    def register_instance(self, provider_instance):
        """Registers a pre-instantiated provider instance with custom injected dependencies."""
        self._providers.append(provider_instance)

        # Also register its tools
        if hasattr(provider_instance, "get_tools"):
            self._tools.extend(provider_instance.get_tools())

    def get_all_tools(self) -> List[BaseTool]:
        """
        Returns a flat list of all tools from all registered providers.
        """
        return self._tools

    def get_provider(self, provider_cls: Type[T]) -> Optional[T]:
        """
        Safely retrieves a registered tool provider instance by its class type.
        Returns None if the provider is not found.
        """
        return next(
            (p for p in self._providers if isinstance(p, provider_cls)),
            None
        )