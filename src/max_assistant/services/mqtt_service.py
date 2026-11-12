# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
import asyncio
import json
import logging
from typing import Optional
import aiomqtt

logger = logging.getLogger(__name__)


class MQTTService:
    def __init__(self, broker_host: str, event_bus, broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.event_bus = event_bus
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        """Cancels the background MQTT listener task and waits for a clean teardown."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("MQTTService: Stopped.")

    async def _run(self):
        while True:
            try:
                async with aiomqtt.Client(self.broker_host, port=self.broker_port) as client:
                    await client.subscribe("max/presence/#")
                    logger.info("MQTTService: Subscribed to synthetic presence topics.")
                    async for message in client.messages:
                        await self._dispatch(message)
            except aiomqtt.MqttError as e:
                logger.warning(f"MQTTService connection lost: {e}. Retrying in 5s...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break

    async def _dispatch(self, message):
        try:
            payload = json.loads(message.payload.decode())
            # Maps MQTT topic `max/presence/bedroom` -> internal topic `presence.bedroom`
            sub_topic = str(message.topic).replace("max/", "").replace("/", ".")
            await self.event_bus.publish(sub_topic, payload)
        except Exception as e:
            logger.error(f"MQTTService: Failed to parse message: {e}", exc_info=True)