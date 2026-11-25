import asyncio
import json
import aiomqtt
import pytest
from max_assistant.events.bus import LocalEventBus
from max_assistant.services.mqtt_service import MQTTService
from max_assistant.services.presence_cep import PresenceCEPWorker

BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883


@pytest.mark.asyncio
async def test_mqtt_sensor_sequence_triggers_user_not_present():
    event_bus = LocalEventBus()
    received_bus_events = []

    async def on_presence_event(topic: str, data: dict):
        received_bus_events.append((topic, data))

    event_bus.subscribe("presence.*", on_presence_event)

    cep_worker = PresenceCEPWorker(broker_host=BROKER_HOST, broker_port=BROKER_PORT)
    cep_worker.start()

    mqtt_service = MQTTService(broker_host=BROKER_HOST, broker_port=BROKER_PORT, event_bus=event_bus)
    await mqtt_service.start()

    await asyncio.sleep(0.3)

    try:
        # Matches 'resident_leave_room' in cep_config.yaml:
        # Step 1: room="MAIN", sensor="pir", name="door motion", state="active"
        # Step 2: room="main", sensor="door", state="closed"
        async with aiomqtt.Client(BROKER_HOST, port=BROKER_PORT) as pub_client:
            await pub_client.publish(
                "home/MAIN/sensor/pir/door motion",
                json.dumps({"state": "active", "user_id": "Margaret.Miller"})
            )
            await asyncio.sleep(0.1)

            await pub_client.publish(
                "home/main/sensor/door",
                json.dumps({"state": "closed", "user_id": "Margaret.Miller"})
            )

        await asyncio.sleep(0.5)

        assert len(received_bus_events) == 1, (
            f"Expected 1 presence event on bus, got {len(received_bus_events)}. "
            f"CEP sensor history: {cep_worker._sensor_history}"
        )
        topic, payload = received_bus_events[0]
        assert topic == "presence.main"
        assert payload["state"] == "vacated"
        assert payload["user_id"] == "Margaret.Miller"
        assert payload["trigger_rule"] == "resident_leave_room"
    finally:
        await mqtt_service.stop()
        cep_worker.stop()


@pytest.mark.asyncio
async def test_mqtt_sensor_sequence_triggers_resident_enters_room():
    event_bus = LocalEventBus()
    received_bus_events = []

    async def on_presence_event(topic: str, data: dict):
        received_bus_events.append((topic, data))

    event_bus.subscribe("presence.*", on_presence_event)

    cep_worker = PresenceCEPWorker(broker_host=BROKER_HOST, broker_port=BROKER_PORT)
    cep_worker.start()

    mqtt_service = MQTTService(broker_host=BROKER_HOST, broker_port=BROKER_PORT, event_bus=event_bus)
    await mqtt_service.start()

    await asyncio.sleep(0.3)

    try:
        # Matches 'resident_enter_room' in cep_config.yaml:
        # Step 1: room="main", sensor="door", state="open"
        # Step 2: room="main", sensor="pir", name="door motion", state="active"
        async with aiomqtt.Client(BROKER_HOST, port=BROKER_PORT) as pub_client:
            await pub_client.publish(
                "home/main/sensor/door",
                json.dumps({"state": "open", "user_id": "Margaret.Miller"})
            )
            await asyncio.sleep(0.1)

            await pub_client.publish(
                "home/main/sensor/pir/door motion",
                json.dumps({"state": "active", "user_id": "Margaret.Miller"})
            )

        await asyncio.sleep(0.5)

        assert len(received_bus_events) == 1, (
            f"Expected 1 presence event on bus, got {len(received_bus_events)}. "
            f"CEP sensor history: {cep_worker._sensor_history}"
        )
        topic, payload = received_bus_events[0]
        assert topic == "presence.main"
        assert payload["state"] == "occupied"
        assert payload["user_id"] == "Margaret.Miller"
        assert payload["trigger_rule"] == "resident_enter_room"
    finally:
        await mqtt_service.stop()
        cep_worker.stop()


@pytest.mark.asyncio
async def test_mqtt_presence_transitions_from_vacated_to_occupied():
    event_bus = LocalEventBus()
    received_bus_events = []

    async def on_presence_event(topic: str, data: dict):
        received_bus_events.append((topic, data))

    event_bus.subscribe("presence.*", on_presence_event)

    cep_worker = PresenceCEPWorker(broker_host=BROKER_HOST, broker_port=BROKER_PORT)
    cep_worker.start()

    mqtt_service = MQTTService(broker_host=BROKER_HOST, broker_port=BROKER_PORT, event_bus=event_bus)
    await mqtt_service.start()

    await asyncio.sleep(0.3)

    try:
        async with aiomqtt.Client(BROKER_HOST, port=BROKER_PORT) as pub_client:
            # Phase 1: Resident leaves the room -> initially out of the room ("vacated")
            await pub_client.publish(
                "home/MAIN/sensor/pir/door motion",
                json.dumps({"state": "active", "user_id": "Margaret.Miller"})
            )
            await asyncio.sleep(0.1)

            await pub_client.publish(
                "home/main/sensor/door",
                json.dumps({"state": "closed", "user_id": "Margaret.Miller"})
            )
            await asyncio.sleep(0.4)

            assert len(received_bus_events) == 1
            assert received_bus_events[0][0] == "presence.main"
            assert received_bus_events[0][1]["state"] == "vacated"
            assert received_bus_events[0][1]["trigger_rule"] == "resident_leave_room"

            # Phase 2: Resident enters the room -> transitions to ("occupied")
            await pub_client.publish(
                "home/main/sensor/door",
                json.dumps({"state": "open", "user_id": "Margaret.Miller"})
            )
            await asyncio.sleep(0.1)

            await pub_client.publish(
                "home/main/sensor/pir/door motion",
                json.dumps({"state": "active", "user_id": "Margaret.Miller"})
            )
            await asyncio.sleep(0.4)

        assert len(received_bus_events) == 2, (
            f"Expected 2 presence events (vacated then occupied), got {len(received_bus_events)}: "
            f"{received_bus_events}"
        )
        topic, payload = received_bus_events[1]
        assert topic == "presence.main"
        assert payload["state"] == "occupied"
        assert payload["user_id"] == "Margaret.Miller"
        assert payload["trigger_rule"] == "resident_enter_room"
    finally:
        await mqtt_service.stop()
        cep_worker.stop()