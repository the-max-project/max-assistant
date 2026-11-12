# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
import logging

logger = logging.getLogger(__name__)

class NotificationRouter:
    """Subscribes to internal bus events, updates DB state, and alerts active user sessions."""

    def __init__(self, event_bus, session_mgr, db_client):
        self.event_bus = event_bus
        self.session_mgr = session_mgr
        self.db_client = db_client

        # Register event handlers
        self.event_bus.subscribe("task.due", self.on_task_due)
        self.event_bus.subscribe("presence.*", self.on_presence_change)

    async def on_task_due(self, _topic: str, data: dict):
        user_id = data.get("user_id")
        if not user_id or not isinstance(user_id, str):
            logger.warning("on_task_due received event without a valid user_id. Discarding.")
            return

        payload = {"task_id": data.get("task_id"), "text": data.get("text")}
        await self._broadcast_to_user(user_id, payload)

    async def on_presence_change(self, _topic: str, data: dict):
        room = data.get("room")
        state = data.get("state")
        user_id = data.get("user_id", "default_user")

        if state == "vacated":
            # 1. Update Current Room in Neo4j
            # language=Cypher
            query_update = """
            MATCH (u:User {id: $user_id})
            MERGE (h:Room {name: "hallway"})
            OPTIONAL MATCH (u)-[curr:CURRENTLY_IN]->(:Room)
            DELETE curr
            MERGE (u)-[:CURRENTLY_IN]->(h)
            """
            await self.db_client.execute_query(query_update, {"user_id": user_id})

            # 2. Check for Context Reminders waiting on leaving this room
            # language=Cypher
            query_tasks = """
            MATCH (u:User {id: $user_id})-[:HAS_TASK]->(t:Task)-[:TRIGGERS_ON]->(trig:EventTrigger)
            WHERE trig.type = 'ON_LEAVE' AND trig.room = $room AND t.status = 'PENDING'
            SET t.status = 'COMPLETED'
            RETURN t.id AS id, t.text AS text
            """
            res = await self.db_client.execute_query(query_tasks, {"user_id": user_id, "room": room})
            for record in res.get("data", []):
                payload = {
                    "task_id": record["id"],
                    "text": f"You just left the {room}. Reminder: {record['text']}"
                }
                await self._broadcast_to_user(user_id, payload)

    async def _broadcast_to_user(self, user_id: str, payload: dict):
        """Pushes events into active WebSocket connections via ConnectionManager[cite: 1, 4]."""
        active_agents = self.session_mgr.get_agents(user_id)
        if not active_agents:
            logger.warning(f"No active session found for user {user_id}. Event held.")
            return

        for agent in active_agents:
            if agent.connection_manager:
                await agent.connection_manager.submit_external_event(payload)