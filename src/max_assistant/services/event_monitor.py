# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Diagnostic subscriber that logs all LocalEventBus events to the
'max_assistant.event_monitor' logger configured in log_config.json.
"""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict

from max_assistant.events.bus import LocalEventBus

logger = logging.getLogger(__name__)
event_logger = logging.getLogger("max_assistant.event_monitor")


class EventMonitor:
    """Subscribes to LocalEventBus and writes structured JSON lines when enabled in log_config.json."""

    def __init__(self, event_bus: LocalEventBus):
        self.event_bus = event_bus
        if event_logger.isEnabledFor(logging.INFO):
            Path("logs").mkdir(parents=True, exist_ok=True)
        self.event_bus.subscribe("*", self._on_event)
        logger.info(
            "EventMonitor subscribed (active=%s via 'max_assistant.event_monitor' log level).",
            event_logger.isEnabledFor(logging.INFO),
        )

    def stop(self) -> None:
        """Unsubscribes from LocalEventBus."""
        self.event_bus.unsubscribe("*", self._on_event)
        logger.info("EventMonitor unsubscribed.")

    @staticmethod
    async def _on_event(topic: str, data: Dict[str, Any]) -> None:
        if not event_logger.isEnabledFor(logging.INFO):
            return

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "data": data,
        }
        event_logger.info(json.dumps(record, default=str))