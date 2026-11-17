import json
import logging
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from max_assistant.events.bus import LocalEventBus
from max_assistant.services.event_monitor import EventMonitor


@pytest.mark.asyncio
async def test_monitor_logs_bus_events_and_stops():
    bus = LocalEventBus()
    timestamp = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    with patch('max_assistant.services.event_monitor.event_logger') as event_logger, \
            patch('max_assistant.services.event_monitor.Path') as path, \
            patch('max_assistant.services.event_monitor.datetime') as clock:
        event_logger.isEnabledFor.return_value = True
        clock.now.return_value = timestamp
        monitor = EventMonitor(bus)

        path.assert_called_once_with('logs')
        path.return_value.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        await bus.publish('presence.bedroom', {'when': timestamp, 'state': 'occupied'})

        clock.now.assert_called_once_with(timezone.utc)
        event_logger.info.assert_called_once()
        record = json.loads(event_logger.info.call_args.args[0])
        assert record == {
            'ts': timestamp.isoformat(),
            'topic': 'presence.bedroom',
            'data': {'when': str(timestamp), 'state': 'occupied'},
        }

        monitor.stop()
        await bus.publish('presence.bedroom', {'state': 'vacated'})
        event_logger.info.assert_called_once()


@pytest.mark.asyncio
async def test_monitor_skips_logging_when_disabled_and_can_be_enabled_later():
    bus = LocalEventBus()
    with patch('max_assistant.services.event_monitor.event_logger') as event_logger, \
            patch('max_assistant.services.event_monitor.Path') as path:
        event_logger.isEnabledFor.return_value = False
        monitor = EventMonitor(bus)
        path.assert_not_called()

        await bus.publish('task.due', {'id': 1})
        event_logger.info.assert_not_called()
        event_logger.isEnabledFor.assert_called_with(logging.INFO)

        event_logger.isEnabledFor.return_value = True
        await bus.publish('task.due', {'id': 2})
        assert json.loads(event_logger.info.call_args.args[0])['data'] == {'id': 2}
        monitor.stop()