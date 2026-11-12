# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
import asyncio
import json
import logging
import aiomqtt

logger = logging.getLogger(__name__)

class MQTTService:
    def __init__(self, broker_host: str, event_bus, broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.event_bus = event_bus

    async def start(self):
        asyncio.create_task(self._run())

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