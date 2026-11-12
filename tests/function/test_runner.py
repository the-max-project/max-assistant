import asyncio
import inspect
import json
import time
import unicodedata
from typing import List, Dict, Any, Optional
from urllib import request

from unittest.mock import AsyncMock, MagicMock

from max_assistant.app_services import AppServices
from max_assistant.agent.session_mgr import session_manager
from max_assistant.agent.agent import Agent
from max_assistant.config import OLLAMA_BASE_URL
from max_assistant.tools import PersonTools
from tests.function.types import ScenarioResult, StepResult
from tests.function.conftest import SEMANTIC_VALIDATOR_MODEL

# Global tracker for the model currently resident in memory
_ACTIVE_MODEL: Optional[str] = None


async def unload_ollama_model(model_name: str, base_url: str = OLLAMA_BASE_URL) -> None:
    """Sends keep_alive: 0 to Ollama's generate API to immediately evict the model from VRAM."""
    def _post_unload():
        endpoint = f"{base_url.rstrip('/')}/api/generate"
        payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
        req = request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=10.0) as resp:
                resp.read()
            print(f"[OLLAMA] Successfully evicted model '{model_name}' from VRAM.")
        except Exception as exc:
            print(f"[OLLAMA WARN] Failed to evict model '{model_name}': {exc}")

    await asyncio.to_thread(_post_unload)


async def execute_scenario_workflow(
        username: str,
        steps: List[Dict[str, Any]],
        model_name: str,
        scenario_request=None,
        results: Optional[ScenarioResult] = None,

):
    """
    A test execution engine for chat scenarios defined in an injection array.
    Supports both synchronous token validators and asynchronous semantic/graph validators.
    """
    global _ACTIVE_MODEL

    previous_model = _ACTIVE_MODEL

    # Evict the prior model when the model changes across scenario runs
    if previous_model is not None and _ACTIVE_MODEL != model_name:
        print(f"\n[MODEL ROTATION] Switching from '{previous_model}' to '{model_name}'. Unloading previous model...")
        await unload_ollama_model(previous_model)

    _ACTIVE_MODEL = model_name

    if scenario_request is not None and hasattr(scenario_request, "node"):
        testcase_name = scenario_request.node.name
        app_services = await AppServices.create(model_name=model_name)
    else:
        print("No request or node found, skipping AppServices creation.")
        return

    execution_times: List[float ] = []
    responses: List[str] = []
    success = False
    thread_id = None
    step_results: List[StepResult] = []
    agent = None
    user_id = None

    try:
        if not app_services.llm_ready_event.is_set():
            await app_services.llm_ready_event.wait()

        person_tools = PersonTools(app_services.db_client)
        user_data = await person_tools.get_user_info_internal(username)
        if "error" in user_data:
            user_data = {}

        agent = Agent(app_services.reasoning_engine, user_data)
        thread_id = agent.get_thread_id()

        # Wire a spy connection_manager & register with session_manager
        # so NotificationRouter can dispatch pushed events during tests
        mock_conn_mgr = MagicMock()
        mock_conn_mgr.submit_external_event = AsyncMock()
        agent.connection_manager = mock_conn_mgr

        user_id = user_data.get("id", username)
        session_manager.register(user_id, agent)  # or session_manager.add_agent(user_id, agent)

        # This creates a highly scannable visual header inside PyCharm's console window
        print("\n" + "=" * 80)
        print(f" WORKING THREAD ID : {thread_id}")
        print(f" USERNAME          : {username}")
        print(f" OLLAMA MODEL      : {model_name}")
        print(f" Testcase          : {testcase_name}")
        print("=" * 80 + "\n")

    # Execute conversational array step-by-step

        for index, step in enumerate(steps, start=1):
            validators = step.get("validators", [])
            start_time = time.perf_counter()

            # --- BRANCH A: Direct Event Bus Injection ---
            if "inject_event" in step:
                event_cfg = step["inject_event"]
                topic = event_cfg["topic"]
                payload = dict(event_cfg.get("payload", {}))
                # Default user_id to the scenario's resolved user if omitted
                payload.setdefault("user_id", user_id)
                settle_time = event_cfg.get("settle_time", 0.1)

                # Reset mock before injecting so we only capture events from this step
                mock_conn_mgr.submit_external_event.reset_mock()

                await app_services.event_bus.inject_test_event(
                    topic=topic,
                    payload=payload,
                    settle_time=settle_time,
                )

                # Extract pushed payload (or pass through agent.handle_push_event if testing full LLM response)
                if mock_conn_mgr.submit_external_event.called:
                    pushed_payload = mock_conn_mgr.submit_external_event.call_args[0][0]
                    # Option 1: Validate the raw pushed text directly:
                    actual_response = pushed_payload.get("text", json.dumps(pushed_payload))
                    # Option 2: Or feed it through the agent's push handler to test the LLM's spoken alert:
                    # actual_response = await agent.handle_push_event(pushed_payload)
                else:
                    actual_response = ""

                step_label = f"[EVENT INJECT: {topic}]"

            # --- BRANCH B: Standard User Chat Input ---
            elif "user_input" in step:
                user_input = step["user_input"]
                if isinstance(user_input, str):
                    user_input = unicodedata.normalize("NFKC", user_input)
                actual_response = await agent.ainvoke(user_input)
                step_label = user_input
            else:
                raise ValueError(
                    f"Step {index} in testcase '{testcase_name}' must contain 'user_input' or 'inject_event'."
                )

            step_duration = time.perf_counter() - start_time
            execution_times.append(step_duration)

            if isinstance(actual_response, str):
                actual_response = unicodedata.normalize("NFKC", actual_response)
            responses.append(actual_response)

            # Run existing validators unchanged
            for validator_fn in validators:
                sig = inspect.signature(validator_fn)
                kwargs = {}
                if "model_name" in sig.parameters:
                    kwargs["model_name"] = SEMANTIC_VALIDATOR_MODEL or model_name

                if inspect.iscoroutinefunction(validator_fn):
                    await validator_fn(actual_response, app_services.db_client, **kwargs)
                else:
                    validator_fn(actual_response, app_services.db_client, **kwargs)

            step_results.append(StepResult(
                step=index,
                user_input=step_label,
                actual_result=actual_response,
                elapsed_time=step_duration
            ))

        success = True

    finally:
        if agent and user_id:
            session_manager.unregister(user_id, agent)

        # Guarantee safe database connection pooling teardown
        if app_services.db_client:
            await app_services.db_client.close()

        if results is not None:
            results["testcase_name"] = testcase_name
            results["thread_id"] = thread_id
            results["model"] = model_name
            results["success"] = success
            results["total_execution_time"] = sum(execution_times) if execution_times else 0
            results["step_results"] = step_results

        # Print the execution time metrics at the end of the scenario
        if execution_times:
            total_time = sum(execution_times)
            avg_time = total_time / len(execution_times) if execution_times else 0

            print("\n" + "=" * 80)
            print(f" PERFORMANCE SUMMARY")
            print(f" Total ainvoke calls : {len(execution_times)}")
            print(f" Total Execution Time: {total_time:.4f} seconds")
            print(f" Average Time / Step : {avg_time:.4f} seconds")
            print("=" * 80 + "\n")


def get_active_model() -> Optional[str]:
    """Returns the name of the model currently loaded in the runner."""
    global _ACTIVE_MODEL
    return _ACTIVE_MODEL


