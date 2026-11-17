# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines the AppServices container class to centralize service initialization.
Updated to wire the LocalEventBus, PersistentTimeScheduler, and MQTT services.
"""

import logging
import asyncio
import os
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
from max_assistant.tools.reminder_tools import ReminderTools
from max_assistant.agent.graph import create_reasoning_engine
from max_assistant.agent.session_mgr import session_manager

# New Event-Driven & MQTT imports
from max_assistant.events.bus import LocalEventBus
from max_assistant.services.time_scheduler import PersistentTimeScheduler
from max_assistant.services.presence_cep import PresenceCEPWorker
from max_assistant.services.mqtt_service import MQTTService
from max_assistant.services.notification_router import NotificationRouter
from max_assistant.services.event_monitor import EventMonitor
from max_assistant.services.playbook_service import PlaybookService
from max_assistant.services.engagement_manager import EngagementManager
from max_assistant.tools.playbook_tools import PlaybookTools

logger = logging.getLogger(__name__)

ReasoningEngine = RunnableSerializable
T = TypeVar('T')


class AppServices:
    """
    A container class to encapsulate all singleton services for the application.
    Centralizes asynchronous service initialization.

    Do not instantiate directly; use the async factory method `AppServices.create()`.
    """

    def __init__(
            self,
            private_token: str,
            db_client: Neo4jClient,
            llm: ChatOllama,
            tool_registry: ToolRegistry,
            reasoning_engine: ReasoningEngine,
            llm_ready_event: asyncio.Event,
            preload_task: asyncio.Task,
            event_bus: LocalEventBus,
            event_monitor: EventMonitor,
            time_scheduler: PersistentTimeScheduler,
            playbook_service: PlaybookService,
            engagement_manager: EngagementManager,
            mqtt_service: MQTTService,
            cep_worker: PresenceCEPWorker,
            notification_router: NotificationRouter
    ):
        if private_token != "__FACTORY__":
            raise RuntimeError("Use AppServices.create() to instantiate this class.")

        self.db_client = db_client
        self.llm = llm
        self.tool_registry = tool_registry
        self.reasoning_engine = reasoning_engine
        self.llm_ready_event = llm_ready_event
        self._preload_task = preload_task

        # Event-driven infrastructure
        self.event_bus = event_bus
        self.event_monitor = event_monitor
        self.time_scheduler = time_scheduler
        self.playbook_service = playbook_service
        self.engagement_manager = engagement_manager
        self.mqtt_service = mqtt_service
        self.cep_worker = cep_worker
        self.notification_router = notification_router

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

            # 2. Initialize Event Bus and Time Scheduler
            event_bus = LocalEventBus()
            event_monitor = EventMonitor(event_bus=event_bus)
            time_scheduler = PersistentTimeScheduler(db_client=db_client, event_bus=event_bus)
            playbook_service = PlaybookService(
                db_client=db_client,
                llm=llm,
                time_scheduler=time_scheduler,
                event_bus=event_bus,
            )

            # 3. Initialize and Populate Tool Registry (Injecting time_scheduler into providers)
            tool_registry = cls._initialize_tool_registry(
                db_client, llm, time_scheduler, playbook_service
            )

            # 4. Create Reasoning Engine
            reasoning_engine = await create_reasoning_engine(llm, tool_registry)
            logger.info("Reasoning engine initialized.")

            # 5. Initialize MQTT & Router Infrastructure
            mqtt_host = os.getenv("MQTT_BROKER_HOST", "mosquitto")
            mqtt_port = int(os.getenv("MQTT_BROKER_PORT", "1883"))

            mqtt_service = MQTTService(broker_host=mqtt_host, event_bus=event_bus, broker_port=mqtt_port)
            cep_worker = PresenceCEPWorker(broker_host=mqtt_host, broker_port=mqtt_port)
            notification_router = NotificationRouter(
                event_bus=event_bus,
                session_mgr=session_manager,
                db_client=db_client,
                time_scheduler=time_scheduler
            )
            engagement_manager = EngagementManager(
                event_bus=event_bus,
                session_mgr=session_manager,
                db_client=db_client,
                notification_router=notification_router
            )

            # 6. Return the fully configured container instance
            return cls(
                private_token="__FACTORY__",
                db_client=db_client,
                llm=llm,
                tool_registry=tool_registry,
                reasoning_engine=reasoning_engine,
                llm_ready_event=llm_ready_event,
                preload_task=preload_task,
                event_bus=event_bus,
                event_monitor=event_monitor,
                time_scheduler=time_scheduler,
                playbook_service=playbook_service,
                mqtt_service=mqtt_service,
                cep_worker=cep_worker,
                notification_router=notification_router,
                engagement_manager=engagement_manager,
            )

        except Exception as e:
            logger.critical(f"Failed to initialize application services: {e}", exc_info=True)
            raise

    @classmethod
    async def _initialize_clients(
            cls,
            llm_ready_event: asyncio.Event,
            model_name: Optional[str] = None
    ) -> Tuple[Neo4jClient, ChatOllama, asyncio.Task]:
        """Initializes the Neo4j client and the Ollama LLM concurrently."""
        logger.info("Initializing Neo4j client and LLM...")

        active_model = model_name or OLLAMA_MODEL_NAME
        llm = create_llm_instance(active_model, OLLAMA_BASE_URL, temperature=0)

        try:
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
    def _initialize_tool_registry(
            db_client: Neo4jClient,
            llm: ChatOllama,
            time_scheduler: PersistentTimeScheduler,
            playbook_service: PlaybookService
    ) -> ToolRegistry:
        logger.info("Initializing and populating tool registry...")
        tool_registry = ToolRegistry(db_client=db_client, llm=llm)

        for provider_class in ALL_TOOL_PROVIDERS:
            if issubclass(provider_class, ReminderTools) or provider_class is ReminderTools:
                provider_instance = provider_class(
                    db_client=db_client, time_scheduler=time_scheduler, llm=llm
                )
                tool_registry.register_instance(provider_instance)
            elif issubclass(provider_class, PlaybookTools) or provider_class is PlaybookTools:
                provider_instance = provider_class(
                    db_client=db_client, playbook_service=playbook_service, llm=llm
                )
                tool_registry.register_instance(provider_instance)
            else:
                tool_registry.register_provider(provider_class)
            logger.info(f"-> Registered tool provider: {provider_class.__name__}")

        logger.info(f"Tool registry populated with {len(ALL_TOOL_PROVIDERS)} providers.")
        return tool_registry

    def get_tool_provider(self, provider_cls: Type[T]) -> Optional[T]:
        """Delegates the provider lookup to the tool registry."""
        return self.tool_registry.get_provider(provider_cls)