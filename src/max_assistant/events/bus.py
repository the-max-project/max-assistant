# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
import asyncio
from collections import defaultdict
import fnmatch
import logging
from typing import Callable, Coroutine, Dict, Set

logger = logging.getLogger(__name__)

# Conversational & Turn Lifecycle Topics
TOPIC_CHAT_TURN_STARTED = "chat.turn.started"
TOPIC_CHAT_TURN_COMPLETED = "chat.turn.completed"
TOPIC_USER_SPEECH_TRANSCRIBED = "user.speech.transcribed"
TOPIC_AGENT_RESPONSE_GENERATED = "agent.response.generated"

class LocalEventBus:
    """Asynchronous publish-subscribe event bus supporting wildcard routing."""

    def __init__(self):
        self._subscribers: Dict[str, Set[Callable[[str, dict], Coroutine]]] = defaultdict(set)

    def subscribe(self, pattern: str, handler: Callable[[str, dict], Coroutine]):
        """Subscribes an async callback to a topic pattern (e.g., 'task.*', 'presence.*')."""
        self._subscribers[pattern].add(handler)

    def unsubscribe(self, pattern: str, handler: Callable[[str, dict], Coroutine]):
        self._subscribers[pattern].discard(handler)

    async def publish(self, topic: str, data: dict):
        """Asynchronously dispatches an event to all matching subscribers."""
        tasks = []
        for pattern, handlers in self._subscribers.items():
            if fnmatch.fnmatch(topic, pattern):
                for handler in handlers:
                    tasks.append(asyncio.create_task(self._safe_dispatch(handler, topic, data)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _safe_dispatch(handler: Callable[[str, dict], Coroutine], topic: str, data: dict):
        try:
            await handler(topic, data)
        except Exception as e:
            handler_name = getattr(handler, "__name__", repr(handler))
            logger.error(f"EventBus error in {handler_name} on '{topic}': {e}", exc_info=True)

    # --- TESTING HARNESS METHOD ---
    async def inject_test_event(self, topic: str, payload: dict, settle_time: float = 0.05):
        """
        Public injection hook for unit/integration tests.
        Publishes the event and yields control to the event loop so listeners finish executing.
        """
        logger.info(f"[TEST INJECTION] Firing event: {topic} with payload: {payload}")
        await self.publish(topic, payload)
        if settle_time > 0:
            await asyncio.sleep(settle_time)