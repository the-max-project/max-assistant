import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from max_assistant.app_services import AppServices


@pytest.mark.asyncio
async def test_initialize_clients_returns_db_llm_and_preload_task():
    """
    Verifies that _initialize_clients concurrently spins up the Neo4j client
    and safely hands back the active LLM instance and its strong preload task reference.
    """
    fake_db = MagicMock()
    fake_llm = MagicMock()
    fake_task = MagicMock(spec=asyncio.Task)

    # Use a real asyncio.Event since the underlying code interacts with it directly
    ready_event = asyncio.Event()

    with patch("max_assistant.app_services.Neo4jClient.create", new=AsyncMock(return_value=fake_db)), \
            patch("max_assistant.app_services.create_llm_instance", return_value=fake_llm), \
            patch("max_assistant.app_services.AppServices._wrap_preload_task", new=AsyncMock(return_value=fake_task)):
        db_client, llm, preload_task = await AppServices._initialize_clients(ready_event)

    assert db_client is fake_db
    assert llm is fake_llm
    assert preload_task is fake_task


def test_initialize_tool_registry_registers_providers():
    """
    Verifies that the tool registry successfully initializes and dynamically
    iterates through ALL_TOOL_PROVIDERS to mount capabilities.
    """
    db_client = MagicMock()
    llm = MagicMock()
    fake_registry = MagicMock()
    fake_time_scheduler = MagicMock()
    fake_playbook_service = MagicMock()
    fake_event_bus = MagicMock()

    # Create dummy classes with __name__ attributes to satisfy the logger engine
    class MockProvider1: pass

    class MockProvider2: pass

    with patch("max_assistant.app_services.ToolRegistry", return_value=fake_registry), \
            patch("max_assistant.app_services.ALL_TOOL_PROVIDERS", [MockProvider1, MockProvider2]):
        # Pass the fourth argument here
        result = AppServices._initialize_tool_registry(
            db_client,
            llm,
            fake_time_scheduler,
            fake_playbook_service,
            fake_event_bus,
        )

    assert result is fake_registry
    # Confirms that register_provider was invoked for each provider in the tool array
    assert fake_registry.register_provider.call_count == 2


@pytest.mark.asyncio
async def test_app_services_factory_lifecycle_compilation():
    """
    Sanity checks the full orchestrator factory method `AppServices.create()`.
    Ensures that the private token guard works, components are wired,
    and the compiled reasoning engine is successfully generated.
    """
    fake_db = MagicMock()
    fake_llm = MagicMock()
    fake_task = MagicMock(spec=asyncio.Task)
    fake_registry = MagicMock()
    fake_engine = MagicMock()

    with patch("max_assistant.app_services.AppServices._initialize_clients",
               new=AsyncMock(return_value=(fake_db, fake_llm, fake_task))), \
            patch("max_assistant.app_services.AppServices._initialize_tool_registry",
                  return_value=fake_registry), \
            patch("max_assistant.app_services.create_reasoning_engine",
                  new=AsyncMock(return_value=fake_engine)):
        services = await AppServices.create()

    assert services.db_client is fake_db
    assert services.llm is fake_llm
    assert services.tool_registry is fake_registry
    assert services.reasoning_engine is fake_engine
    assert services._preload_task is fake_task


def test_direct_instantiation_guard_raises_runtime_error():
    """
    Ensures that direct instantiation bypassing the `AppServices.create()` 
    asynchronous factory method raises a defensive RuntimeError.
    """
    with pytest.raises(RuntimeError) as exc_info:
        AppServices(
            private_token="INVALID_TOKEN",
            db_client=MagicMock(),
            llm=MagicMock(),
            tool_registry=MagicMock(),
            reasoning_engine=MagicMock(),
            llm_ready_event=MagicMock(),
            preload_task=MagicMock(),
            event_bus=MagicMock(),
            event_monitor=MagicMock(),
            time_scheduler=MagicMock(),
            playbook_service=MagicMock(),
            engagement_manager=MagicMock(),
            mqtt_service=MagicMock(),
            cep_worker=MagicMock(),
            notification_router=MagicMock(),
        )

    assert "Use AppServices.create() to instantiate this class." in str(exc_info.value)