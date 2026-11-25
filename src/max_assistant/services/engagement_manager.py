# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from max_assistant.events.bus import (
    TOPIC_CHAT_TURN_COMPLETED,
    TOPIC_USER_ENGAGEMENT_IDLE,
)
from max_assistant.services.playbook_service import PlaybookService

logger = logging.getLogger(__name__)


class EngagementManager:
    """Coordinates attention-aware delivery of daily playbook items and wake-up routines."""

    def __init__(
        self,
        event_bus,
        session_mgr,
        db_client,
        notification_router,
        playbook_service: Optional[PlaybookService] = None,
    ):
        self.event_bus = event_bus
        self.session_mgr = session_mgr
        self.db_client = db_client
        self.notification_router = notification_router
        self.playbook_service = playbook_service

        self.event_bus.subscribe("state.sleep", self.on_sleep_state_change)
        self.event_bus.subscribe(TOPIC_CHAT_TURN_COMPLETED, self.on_chat_turn_completed)
        self.event_bus.subscribe(TOPIC_USER_ENGAGEMENT_IDLE, self.on_user_idle)

        self._briefing_delivered_today: Dict[str, str] = {}

    async def on_sleep_state_change(self, _topic: str, data: dict):
        state = data.get("state")
        raw_user_id = data.get("user_id")

        if state == "awake" and raw_user_id:
            user_id = str(raw_user_id)
            now = datetime.now()
            today = now.date().isoformat()
            if self._briefing_delivered_today.get(user_id) == today:
                return  # Already briefed today

            # Fetch today's playbook summary
            query = """
            MATCH (u:User {id: $user_id})-[:HAS_PLAYBOOK]->(p:Playbook {date: $date})
            RETURN p.morning_summary AS summary
            """
            res = await self.db_client.execute_query(query, {"user_id": user_id, "date": today})
            records = res.get("data", [])

            summary = records[0].get("summary") if records else None

            # Boot catch-up: synthesize today's playbook if service started after the 5:30 AM cron window
            if not summary and self.playbook_service is not None:
                playbook = await self.playbook_service.get_or_create_playbook(
                    user_id=user_id, current_dt=now
                )
                if playbook and playbook.morning_summary:
                    summary = playbook.morning_summary

            if summary:
                self._briefing_delivered_today[user_id] = today
                payload = {
                    "type": "WAKE_UP_BRIEFING",
                    "text": (
                        "PROACTIVE PROMPT DIRECTIVE: The resident just woke up. "
                        "Deliver a calm, warm morning greeting in 2-3 short sentences using this summary. "
                        f"Do NOT call any tools. Summary: {summary}"
                    ),
                }
                await self._broadcast(user_id, payload)

    async def on_chat_turn_completed(self, _topic: str, data: dict):
        """Deliver opportunistic prompts after a user finishes their active chat."""
        if data.get("is_background"):
            return

        raw_user_id = data.get("user_id")
        if not raw_user_id:
            return
        user_id = str(raw_user_id)

        # Slight delay to ensure conversation is truly idle
        await asyncio.sleep(3.0)

        if not self.notification_router.is_user_in_active_turn(user_id):
            await self._evaluate_opportunistic_prompt(user_id)

    async def on_user_idle(self, _topic: str, data: dict):
        """Triggered when the user is present but bored/idle for an extended period."""
        raw_user_id = data.get("user_id")
        if not raw_user_id:
            return
        user_id = str(raw_user_id)

        if not self.notification_router.is_user_in_active_turn(user_id):
            await self._evaluate_opportunistic_prompt(user_id)

    async def _evaluate_opportunistic_prompt(self, user_id: str):
        """Pops the highest priority queued engagement item from the Playbook."""
        logger.info(f"Evaluating opportunistic prompt for user {user_id}")
        now = datetime.now()
        today = now.date().isoformat()

        # Check presence and sleep state first
        state_query = """
        MATCH (u:User {id: $user_id})
        RETURN coalesce(u.presence, 'present') AS presence, coalesce(u.sleep_state, 'awake') AS sleep_state
        """
        state_res = await self.db_client.execute_query(state_query, {"user_id": user_id})
        if not state_res.get("data"):
            logger.info(f"No presence or sleep_state found for user {user_id}, skipping opportunistic prompt")
            return

        u_state = state_res["data"][0]
        if u_state["presence"] != "present" or u_state["sleep_state"] != "awake":
            logger.info(
                f"User {user_id} has state presence = {u_state['presence']} and sleep_state = "
                f"{u_state['sleep_state']}, skipping opportunistic prompt"
            )
            return

        # Fetch candidate queued items ordered by priority
        query = """
        MATCH (u:User {id: $user_id})-[:HAS_PLAYBOOK]->(p:Playbook {date: $date})-[:HAS_ITEM]->(pi:PlaybookItem)
        WHERE pi.status = 'QUEUED' 
          AND pi.category IN ['OPTIONAL_ACTIVITY', 'FAMILY_MESSAGE', 'FAMILY_NEWS']
        RETURN properties(pi) AS item
        ORDER BY pi.priority DESC, coalesce(pi.event_time, '23:59') ASC
        """
        res = await self.db_client.execute_query(query, {"user_id": user_id, "date": today})
        records = res.get("data", [])

        selected_item: Optional[Dict[str, Any]] = None
        expired_ids = []

        for row in records:
            item = row["item"]
            if item["category"] == "OPTIONAL_ACTIVITY" and item.get("event_time"):
                event_dt = PlaybookService.resolve_event_datetime(now.date(), str(item["event_time"]))
                if event_dt is not None:
                    lead_minutes = item.get("lead_minutes") if item.get("lead_minutes") is not None else 15
                    due_dt = event_dt - timedelta(minutes=lead_minutes)
                    if due_dt <= now:
                        expired_ids.append(item["id"])
                        continue
            selected_item = item
            break

        if expired_ids:
            await self.db_client.execute_query(
                "MATCH (pi:PlaybookItem) WHERE pi.id IN $ids SET pi.status = 'EXPIRED', pi.updated_at = datetime($now)",
                {"ids": expired_ids, "now": now.isoformat()},
            )

        if selected_item is None:
            logger.info(f"No playbook item found for user {user_id}, skipping opportunistic prompt")
            return

        await self.db_client.execute_query(
            "MATCH (pi:PlaybookItem {id: $id}) SET pi.status = 'PROMPTED', pi.updated_at = datetime($now)",
            {"id": selected_item["id"], "now": now.isoformat()},
        )

        directive = ""

        if selected_item["category"] == "OPTIONAL_ACTIVITY":
            directive = (
                f"PROACTIVE PROMPT DIRECTIVE: The resident is currently idle. "
                f"In 1-2 warm, simple sentences, ask if they would like to attend "
                f"'{selected_item['title']}' and if they want a reminder set. "
                f"Do NOT call any tools right now—wait for their reply. "
                f"[Internal metadata for next turn only—do not say aloud: item_id='{selected_item['id']}']"
            )
        elif selected_item["category"] == "FAMILY_MESSAGE":
            directive = (
                f"PROACTIVE PROMPT DIRECTIVE: Share this family message warmly "
                f"in 1-2 sentences: {selected_item.get('sender', 'Family')} wrote '{selected_item['summary']}'. "
                f"Do NOT call any tools."
            )
        elif selected_item["category"] == "FAMILY_NEWS":
            directive = (
                f"PROACTIVE PROMPT DIRECTIVE: Share this family update warmly "
                f"in 1-2 sentences to spark light conversation: '{selected_item['summary']}'. "
                f"Do NOT call any tools."
            )

        if directive:
            payload = {
                "type": "PLAYBOOK_PROMPT",
                "item_id": selected_item["id"],
                "text": directive,
            }
            await self._broadcast(user_id, payload)

    async def _broadcast(self, user_id: str, payload: dict):
        active_agents = self.session_mgr.get_agents(user_id)
        for agent in active_agents:
            if agent.connection_manager:
                await agent.connection_manager.submit_external_event(payload)