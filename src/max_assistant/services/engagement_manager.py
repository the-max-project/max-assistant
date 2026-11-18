# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
import asyncio
import logging
from datetime import datetime

from max_assistant.events.bus import (
    TOPIC_CHAT_TURN_COMPLETED,
    TOPIC_USER_ENGAGEMENT_IDLE
)

logger = logging.getLogger(__name__)


class EngagementManager:
    """Coordinates attention-aware delivery of daily playbook items and wake-up routines."""

    def __init__(self, event_bus, session_mgr, db_client, notification_router):
        self.event_bus = event_bus
        self.session_mgr = session_mgr
        self.db_client = db_client
        self.notification_router = notification_router

        self.event_bus.subscribe("state.sleep", self.on_sleep_state_change)
        self.event_bus.subscribe(TOPIC_CHAT_TURN_COMPLETED, self.on_chat_turn_completed)
        self.event_bus.subscribe(TOPIC_USER_ENGAGEMENT_IDLE, self.on_user_idle)

        self._briefing_delivered_today = {}

    async def on_sleep_state_change(self, _topic: str, data: dict):
        state = data.get("state")
        user_id = data.get("user_id")

        if state == "awake":
            today = datetime.now().date().isoformat()
            if self._briefing_delivered_today.get(user_id) == today:
                return  # Already briefed today

            # Fetch today's playbook summary
            query = """
            MATCH (u:User {id: $user_id})-[:HAS_PLAYBOOK]->(p:Playbook {date: $date})
            RETURN p.morning_summary AS summary
            """
            res = await self.db_client.execute_query(query, {"user_id": user_id, "date": today})
            records = res.get("data", [])

            if records and records[0].get("summary"):
                self._briefing_delivered_today[user_id] = today
                payload = {
                    "type": "WAKE_UP_BRIEFING",
                    "text": (
                        "PROACTIVE PROMPT DIRECTIVE: The resident just woke up. "
                        "Deliver a calm, warm morning greeting in 2-3 short sentences using this summary. "
                        f"Do NOT call any tools. Summary: {records[0]['summary']}"
                    ),
                }
                await self._broadcast(user_id, payload)

    async def on_chat_turn_completed(self, _topic: str, data: dict):
        """Deliver opportunistic prompts after a user finishes their active chat."""
        user_id = data.get("user_id")

        # Slight delay to ensure conversation is truly idle
        await asyncio.sleep(3.0)

        if not self.notification_router.is_user_in_active_turn(user_id):
            await self._evaluate_opportunistic_prompt(user_id)

    async def on_user_idle(self, _topic: str, data: dict):
        """Triggered when user is present but bored/idle for an extended period."""
        user_id = data.get("user_id")
        if not self.notification_router.is_user_in_active_turn(user_id):
            await self._evaluate_opportunistic_prompt(user_id)

    async def _evaluate_opportunistic_prompt(self, user_id: str):
        """Pops the highest priority queued engagement item from the Playbook."""
        logger.info(f"Evaluating opportunistic prompt for user {user_id}")
        today = datetime.now().date().isoformat()

        # Check presence and sleep state first
        state_query = """
        MATCH (u:User {id: $user_id})
        RETURN coalesce(u.presence, 'present') AS presence, coalesce(u.sleep_state, 'awake') AS sleep_state
        """
        state_res = await self.db_client.execute_query(state_query, {"user_id": user_id})
        if not state_res.get("data"):
            logger.info(f"User {user_id} no presence or sleep_state found, skipping opportunistic prompt")
            return

        u_state = state_res["data"][0]
        if u_state["presence"] != "present" or u_state["sleep_state"] != "awake":
            logger.info(f"User {user_id} has state presence = {u_state['presence']} and sleep_state = "
                        f"{u_state['sleep_state']}, skipping opportunistic prompt")
            return

        # Fetch highest priority queued item
        query = """
        MATCH (u:User {id: $user_id})-[:HAS_PLAYBOOK]->(p:Playbook {date: $date})-[:HAS_ITEM]->(pi:PlaybookItem)
        WHERE pi.status = 'QUEUED' 
          AND pi.category IN ['OPTIONAL_ACTIVITY', 'FAMILY_MESSAGE', 'FAMILY_NEWS']
        WITH pi ORDER BY pi.priority DESC LIMIT 1
        SET pi.status = 'PROMPTED', pi.updated_at = datetime()
        RETURN properties(pi) AS item
        """
        res = await self.db_client.execute_query(query, {"user_id": user_id, "date": today})
        records = res.get("data", [])

        if records:
            item = records[0]["item"]
            directive = ""

            if item["category"] == "OPTIONAL_ACTIVITY":
                directive = (
                    f"PROACTIVE PROMPT DIRECTIVE: The resident is currently idle. "
                    f"In 1-2 warm, simple sentences, ask if they would like to attend "
                    f"'{item['title']}' and if they want a reminder set. "
                    f"Do NOT call any tools right now—wait for their reply. "
                    f"[Internal metadata for next turn only—do not say aloud: item_id='{item['id']}']"
                )
            elif item["category"] == "FAMILY_MESSAGE":
                directive = (
                    f"PROACTIVE PROMPT DIRECTIVE: Share this family message warmly "
                    f"in 1-2 sentences: {item.get('sender', 'Family')} wrote '{item['summary']}'. "
                    f"Do NOT call any tools."
                )
            elif item["category"] == "FAMILY_NEWS":
                directive = (
                    f"PROACTIVE PROMPT DIRECTIVE: Share this family update warmly "
                    f"in 1-2 sentences to spark light conversation: '{item['summary']}'. "
                    f"Do NOT call any tools."
                )

            if directive:
                payload = {
                    "type": "PLAYBOOK_PROMPT",
                    "item_id": item["id"],
                    "text": directive,
                }
                await self._broadcast(user_id, payload)
        else:
            logger.info(f"No playbook item found for user {user_id}, skipping opportunistic prompt")

    async def _broadcast(self, user_id: str, payload: dict):
        active_agents = self.session_mgr.get_agents(user_id)
        for agent in active_agents:
            if agent.connection_manager:
                await agent.connection_manager.submit_external_event(payload)