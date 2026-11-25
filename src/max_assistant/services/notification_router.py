# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
import logging
from typing import Optional, Set

from max_assistant.events.bus import (
    TOPIC_CHAT_TURN_COMPLETED,
    TOPIC_CHAT_TURN_STARTED,
    TOPIC_TASK_DUE,
    TOPIC_PRESENCE_ALL,
    TOPIC_STATE_SLEEP,
    TOPIC_PRESENCE_ALL
)

logger = logging.getLogger(__name__)


class NotificationRouter:
    """Subscribes to internal bus events, updates DB state, and alerts active user sessions."""

    def __init__(self, event_bus, session_mgr, db_client, time_scheduler=None):
        self.event_bus = event_bus
        self.session_mgr = session_mgr
        self.db_client = db_client
        self.time_scheduler = time_scheduler
        self._active_users: Set[str] = set()

        # Register event handlers
        self.event_bus.subscribe(TOPIC_TASK_DUE, self.on_task_due)
        self.event_bus.subscribe(TOPIC_PRESENCE_ALL, self.on_presence_change)
        self.event_bus.subscribe(TOPIC_STATE_SLEEP, self.on_sleep_state_change)
        self.event_bus.subscribe(TOPIC_CHAT_TURN_STARTED, self.on_chat_turn_started)
        self.event_bus.subscribe(TOPIC_CHAT_TURN_COMPLETED, self.on_chat_turn_completed)

    def _wake_scheduler(self) -> None:
        if self.time_scheduler is not None and hasattr(self.time_scheduler, "wake"):
            self.time_scheduler.wake()

    async def on_chat_turn_started(self, _topic: str, data: dict):
        user_id = data.get("user_id")
        if user_id:
            self._active_users.add(str(user_id))

    async def on_chat_turn_completed(self, _topic: str, data: dict):
        user_id = data.get("user_id")
        if user_id:
            self._active_users.discard(str(user_id))

    def is_user_in_active_turn(self, user_id: str) -> bool:
        """Returns True if the user currently has an in-flight LLM/TTS turn."""
        return str(user_id) in self._active_users

    async def on_task_due(self, _topic: str, data: dict):
        user_id = data.get("user_id")
        if not user_id or not isinstance(user_id, str):
            logger.warning("on_task_due received event without a valid user_id. Discarding.")
            return

        payload = {
            "type": "REMINDER",
            "task_id": data.get("task_id"),
            "text": data.get("text"),
        }
        await self._broadcast_to_user(user_id, payload)

    async def on_presence_change(self, _topic: str, data: dict):
        room = data.get("room", "main")
        state = data.get("state")
        scope = data.get("scope", "premise")
        user_id = data.get("user_id", "default_user")

        if state == "occupied":
            # User is inside an active agent room -> mark present and update CURRENTLY_IN
            # language=Cypher
            query_occupied = """
                             MATCH (u:User {id: $user_id})
                             SET u.presence = 'present',
                             u.presence_updated = datetime()
                             MERGE (r:Room {name: $room})
                             OPTIONAL MATCH (u)-[curr:CURRENTLY_IN]->(:Room)
                             DELETE curr
                             MERGE (u)-[:CURRENTLY_IN]->(r) \
                             """
            await self.db_client.execute_query(query_occupied, {"user_id": user_id, "room": room})
            self._wake_scheduler()

        elif state == "vacated":
            if scope == "premise":
                # Resident left the care-home room or exited the house -> effectively OUT / away
                out_location = data.get("out_location", "out")
                # language=Cypher
                query_out = """
                            MATCH (u:User {id: $user_id})
                            SET u.presence = 'away',
                            u.presence_updated = datetime()
                            MERGE (loc:Room {name: $out_location})
                            OPTIONAL MATCH (u)-[curr:CURRENTLY_IN]->(:Room)
                            DELETE curr
                            MERGE (u)-[:CURRENTLY_IN]->(loc) \
                            """
                await self.db_client.execute_query(
                    query_out, {"user_id": user_id, "out_location": out_location}
                )
            else:
                # Multi-room house: moved to an adjacent interior zone (still present in home)
                next_room = data.get("next_room", "hallway")
                # language=Cypher
                query_room_move = """
                                  MATCH (u:User {id: $user_id})
                                  SET u.presence = 'present',
                                  u.presence_updated = datetime()
                                  MERGE (r:Room {name: $next_room})
                                  OPTIONAL MATCH (u)-[curr:CURRENTLY_IN]->(:Room)
                                  DELETE curr
                                  MERGE (u)-[:CURRENTLY_IN]->(r) \
                                  """
                await self.db_client.execute_query(
                    query_room_move, {"user_id": user_id, "next_room": next_room}
                )

            # Evaluate any ON_LEAVE context reminders for the vacated room
            # language=Cypher
            query_tasks = """
                          MATCH (u:User {id: $user_id})-[:HAS_TASK]->(t:Task)-[:TRIGGERS_ON]->(trig:EventTrigger)
                          WHERE trig.type = 'ON_LEAVE' AND trig.room = $room AND t.status = 'PENDING'
                          SET t.status = 'COMPLETED'
                          RETURN t.id AS id, t.text AS text \
                          """
            res = await self.db_client.execute_query(query_tasks, {"user_id": user_id, "room": room})
            for record in res.get("data", []):
                payload = {
                    "type": "REMINDER",
                    "task_id": record["id"],
                    "text": f"The resident just left the {room}. Gently deliver this reminder: {record['text']}",
                }
                await self._broadcast_to_user(user_id, payload)

    async def on_sleep_state_change(self, _topic: str, data: dict):
        state = data.get("state")
        user_id = data.get("user_id", "default_user")

        if state not in ("awake", "asleep"):
            logger.warning(f"on_sleep_state_change received unknown state '{state}'. Ignoring.")
            return

        # language=Cypher
        query = """
        MATCH (u:User {id: $user_id})
        SET u.sleep_state = $state,
            u.sleep_state_updated = datetime()
        """
        await self.db_client.execute_query(query, {"user_id": user_id, "state": state})
        logger.info(f"Updated sleep_state for user '{user_id}' -> '{state}'.")

        if state == "awake":
            # Immediately release any non-wake reminders held overnight
            self._wake_scheduler()

    async def _broadcast_to_user(self, user_id: str, payload: dict):
        """Pushes events into active WebSocket connections via ConnectionManager."""
        active_agents = self.session_mgr.get_agents(user_id)
        if not active_agents:
            logger.warning(f"No active session found for user {user_id}. Event held.")
            return

        for agent in active_agents:
            if agent.connection_manager:
                await agent.connection_manager.submit_external_event(payload)