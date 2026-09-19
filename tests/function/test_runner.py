import asyncio
import inspect
import json
import time
import unicodedata
from typing import List, Dict, Any, Optional
from urllib import request

from max_assistant.app_services import AppServices
from max_assistant.agent.agent import Agent
from max_assistant.config import OLLAMA_BASE_URL
from max_assistant.tools import PersonTools
from tests.function.types import ScenarioResult, StepResult

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
        request=None,
        results: Optional[ScenarioResult] = None,

):
    """
    A test execution engine for chat scenarios defined in an injection array.
    Supports both synchronous token validators and asynchronous semantic/graph validators.
    """
    global _ACTIVE_MODEL

    # Evict the prior model when the model changes across scenario runs
    if _ACTIVE_MODEL is not None and _ACTIVE_MODEL != model_name:
        print(f"\n[MODEL ROTATION] Switching from '{_ACTIVE_MODEL}' to '{model_name}'. Unloading previous model...")
        await unload_ollama_model(_ACTIVE_MODEL)

    _ACTIVE_MODEL = model_name

    testcase_name = "Unknown_Test_Case"
    if request is not None and hasattr(request, "node"):
        testcase_name = request.node.name

        app_services = await AppServices.create(model_name=model_name)
    execution_times: List[float ] = []
    responses: List[str] = []
    success = False
    thread_id = None
    step_results: List[StepResult] = []

    try:
        if not app_services.llm_ready_event.is_set():
            await app_services.llm_ready_event.wait()

        person_tools = PersonTools(app_services.db_client)
        user_data = await person_tools.get_user_info_internal(username)
        if "error" in user_data:
            user_data = {}

        agent = Agent(app_services.reasoning_engine, user_data)
        thread_id = agent.get_thread_id()

        # This creates a highly scannable visual header inside PyCharm's console window
        print("\n" + "=" * 80)
        print(f" WORKING THREAD ID : {thread_id}")
        print(f" USERNAME          : {username}")
        print(f" OLLAMA MODEL      : {model_name}")
        print(f" Testcase          : {testcase_name}")
        print("=" * 80 + "\n")

    # Execute conversational array step-by-step

        for index, step in enumerate(steps, start=1):

            if "user_input" not in step:
                raise ValueError(f"Step {index} in testcase '{testcase_name}' is missing required 'user_input' key.")

            user_input = step["user_input"]
            validators = step.get("validators", [])

            # Uniform normalization of user input to match LLM standards
            if isinstance(user_input, str):
                user_input = unicodedata.normalize("NFKC", user_input)

            # Track start time using perf_counter for high resolution
            start_time = time.perf_counter()

            # Programmatic code execution bypassing AsyncConsoleReader/sys.stdin
            actual_response = await agent.ainvoke(user_input)

            # Track end time, calculate duration, and append to list
            end_time = time.perf_counter()
            step_duration = end_time - start_time
            execution_times.append(step_duration)

            # Normalize LLM Output
            if isinstance(actual_response, str):
                # Safely converts \u202f, \xa0, etc. into standard spaces " "
                # without destroying newlines (\n)
                actual_response = unicodedata.normalize("NFKC", actual_response)
            responses.append(actual_response)

            # Fire off pluggable validators conditionally
            for validator_fn in validators:
                # use the same model for validation as the one under test
                sig = inspect.signature(validator_fn)
                kwargs = {}
                if "model_name" in sig.parameters:
                    kwargs["model_name"] = model_name

                if inspect.iscoroutinefunction(validator_fn):
                    # Natively await async validators (like semantic evaluations or graph checks)
                    await validator_fn(actual_response, app_services.db_client, **kwargs)
                else:
                    # Execute standard synchronous substring/regex assertions directly
                    validator_fn(actual_response, app_services.db_client, **kwargs)

            step_results.append(StepResult(
                step=index,
                user_input=user_input,
                actual_result=actual_response,
                elapsed_time=step_duration
            ))

        success = True

    finally:
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



