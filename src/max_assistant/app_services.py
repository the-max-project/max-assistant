# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines the AppServices container class to centralize service initialization.
Common to main.py and text_client.py.
"""

import logging
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSerializable
from typing import Tuple, Type, TypeVar, Optional

from max_assistant.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    OLLAMA_MODEL_NAME, OLLAMA_BASE_URL
)
from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.clients.ollama_preloader import create_llm_instance, preload_model_async
from max_assistant.tools import ALL_TOOL_PROVIDERS
from max_assistant.tools.registry import ToolRegistry
from max_assistant.agent.graph import create_reasoning_engine

logger = logging.getLogger(__name__)

ReasoningEngine = RunnableSerializable
T = TypeVar('T')


class AppServices:
    """
    A container class to encapsulate all singleton services for the application.
    Centralizes asynchronous service initialization.

    Do not instantiate directly; use the async factory method `AppServices.create()`.
    """

    def __init__(self, private_token, db_client, llm, tool_registry, reasoning_engine, llm_ready_event, preload_task):
        # Defensively prevent direct instantiation bypassing .create()
        if private_token != "__FACTORY__":
            raise RuntimeError("Use AppServices.create() to instantiate this class.")

        self.db_client = db_client
        self.llm = llm
        self.tool_registry = tool_registry
        self.reasoning_engine = reasoning_engine
        self.llm_ready_event = llm_ready_event

        # FIX: Keep a strong reference to prevent garbage collection mid-execution
        self._preload_task = preload_task

    @classmethod
    async def create(cls, model_name: Optional[str] = None) -> "AppServices":
        """
        Asynchronously creates and initializes all application services.
        This is the single source of truth for service setup.
        """
        logger.info("Initializing application services...")
        try:
            llm_ready_event = asyncio.Event()

            # 1. Initialize Core Clients (Concurrently)
            db_client, llm, preload_task = await cls._initialize_clients(llm_ready_event, model_name)

            # 2. Initialize and Populate Tool Registry
            tool_registry = cls._initialize_tool_registry(db_client, llm)

            # 3. Create Reasoning Engine
            reasoning_engine = await create_reasoning_engine(llm, tool_registry)
            logger.info("Reasoning engine initialized.")

            # 4. Return the fully configured container instance
            return cls(
                private_token="__FACTORY__",
                db_client=db_client,
                llm=llm,
                tool_registry=tool_registry,
                reasoning_engine=reasoning_engine,
                llm_ready_event=llm_ready_event,
                preload_task=preload_task
            )

        except Exception as e:
            logger.critical(f"Failed to initialize application services: {e}", exc_info=True)
            raise

    # FIX: Change from @staticmethod to @classmethod so 'cls' is valid
    @classmethod
    async def _initialize_clients(
            cls,
            llm_ready_event: asyncio.Event,
            model_name: Optional[str] = None
    ) -> Tuple[Neo4jClient, ChatOllama, asyncio.Task]:
        """Initializes the Neo4j client and the Ollama LLM concurrently."""
        logger.info("Initializing Neo4j client and LLM...")
        
        # Use existing environment variable or fallback to default in config 
        active_model = model_name or OLLAMA_MODEL_NAME
        
        # Pass temperature through directly, let create_llm_instance handle all other options
        llm = create_llm_instance(active_model, OLLAMA_BASE_URL, temperature=0)
        
        try:
            # Now 'cls' is perfectly bound and safe to call
            db_client, preload_task = await asyncio.gather(
                Neo4jClient.create(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD),
                cls._wrap_preload_task(llm, llm_ready_event)
            )
        except Exception as e:
            logger.critical(f"Concurrent bootstrapping failed: {e}", exc_info=True)
            raise
            
        return db_client, llm, preload_task

    @staticmethod
    async def _wrap_preload_task(llm: ChatOllama, ready_event: asyncio.Event) -> asyncio.Task:
        """Helper wrapper to return the strong task reference directly out of gather."""
        return asyncio.create_task(preload_model_async(llm, ready_event=ready_event))

    @staticmethod
    def _initialize_tool_registry(db_client: Neo4jClient, llm: ChatOllama) -> ToolRegistry:
        """Creates the tool registry and dynamically registers all providers."""
        logger.info("Initializing and populating tool registry...")
        tool_registry = ToolRegistry(db_client=db_client, llm=llm)

        for provider_class in ALL_TOOL_PROVIDERS:
            tool_registry.register_provider(provider_class)
            logger.info(f"-> Registered tool provider: {provider_class.__name__}")

        logger.info(f"Tool registry populated with {len(ALL_TOOL_PROVIDERS)} providers.")
        return tool_registry

    def get_tool_provider(self, provider_cls: Type[T]) -> Optional[T]:
        """Delegates the provider lookup to the tool registry."""
        return self.tool_registry.get_provider(provider_cls)