# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
FastAPI entrypoint and application lifecycle manager.
Coordinates startup and shutdown of the event bus, scheduler, and MQTT workers.
"""

import json
import os
import uvicorn
import asyncio
import logging
import logging.config
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket

from max_assistant.config import PORT, HOST
from max_assistant.app_services import AppServices
from max_assistant.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


def setup_logging(config_path='log_config.json'):
    """Loads logging configuration from a JSON file."""
    os.makedirs('logs', exist_ok=True)
    try:
        with open(config_path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Logging setup error ({type(e).__name__}): Fallback to basic configuration.")
        logging.basicConfig(level=logging.INFO)


setup_logging()


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    Manages application startup and graceful shutdown.
    Initializes core services, starts the persistent scheduler and MQTT listeners.
    """
    logger.info("Application startup initiating...")
    fastapi_app.state.app_services = None

    try:
        # Initialize Service Singletons
        services = await AppServices.create()
        fastapi_app.state.app_services = services

        # Start the Persistent Time Scheduler (Priority Heap hydrated from Neo4j)
        logger.info("Starting PersistentTimeScheduler...")
        await services.time_scheduler.start()

        # Start the Background CEP Worker Thread (Raw MQTT -> Synthetic state)
        logger.info("Starting PresenceCEPWorker background thread...")
        services.cep_worker.start()

        # Start the Main Loop MQTT Ingestion Service (Synthetic state -> EventBus)
        logger.info("Starting MQTTService subscriber...")
        await services.mqtt_service.start()

    except Exception as e:
        logger.critical(f"CRITICAL: Application initialization crashed: {e}", exc_info=True)
        await asyncio.sleep(0.1)
        sys.exit(1)

    yield  # --- Application Execution Phase ---

    logger.info("Application shutting down, executing resource cleanup...")
    services: AppServices | None = fastapi_app.state.app_services

    if services:
        # Stop the Main Loop MQTT Ingestion Service
        logger.info("Stopping MQTTService subscriber...")
        try:
            await services.mqtt_service.stop()
        except Exception as e:
            logger.error(f"Error stopping MQTT service: {e}", exc_info=True)

        # Signal the Background CEP Worker Thread to Stop
        if services.cep_worker:
            logger.info("Stopping PresenceCEPWorker background thread...")
            services.cep_worker.stop()

        # Stop the Persistent Time Scheduler
        logger.info("Stopping PersistentTimeScheduler...")
        try:
            await services.time_scheduler.stop()
        except Exception as e:
            logger.error(f"Error stopping time scheduler: {e}", exc_info=True)

        # Stop EventMonitor after all producers have stopped so trailing events are captured
        if services.event_monitor:
            services.event_monitor.stop()

        # Defensive Database Client Termination - after all other services have stopped
        if services.db_client:
            logger.info("Closing active Neo4j client connection pooling...")
            try:
                await services.db_client.close()
            except Exception as e:
                logger.error(f"Error closing Neo4j connectivity pool safely: {e}", exc_info=True)

    logger.info("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    """Validates the health and routing accessibility of the server instance."""
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    """Handles WebSocket transport layer upgrades and provisions request contexts."""
    services: AppServices | None = app.state.app_services

    await client_ws.accept()
    logger.info("Incoming WebSocket connection accepted.")

    if services is None or not services.reasoning_engine:
        logger.error("Rejecting connection: Core application singletons are uninitialized.")
        await client_ws.close(code=1011, reason="Server error: Services not initialized.")
        return

    logger.info("Provisioning isolated ConnectionManager infrastructure for client session.")
    manager = ConnectionManager(services, client_ws)

    try:
        await manager.handle_connection()
    except Exception as e:
        logger.error(f"Fatal connection handler trap exception: {e}", exc_info=True)
    finally:
        logger.info("Client WebSocket transport cleanup sequence complete.")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_config=None)