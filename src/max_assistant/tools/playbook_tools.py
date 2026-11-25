# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines LangGraph tools for responding to Playbook items and checking daily plans.
"""
from datetime import datetime, timedelta
import json
import logging
from typing import Annotated

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState

from max_assistant.services.playbook_service import PlaybookService
from max_assistant.tools.base_provider import BaseToolProvider

logger = logging.getLogger(__name__)


class PlaybookTools(BaseToolProvider):
    def __init__(self, db_client, playbook_service: PlaybookService, llm=None):
        super().__init__(db_client, llm)
        self.playbook_service = playbook_service

    async def get_remaining_daily_plans(
            self,
            user_info: Annotated[dict, InjectedState("userinfo")],
    ) -> str:
        """
        Use this tool when the user asks about their schedule, plans, or what is happening today.
        It reads today's daily playbook without overwriting accepted activities, filters out events
        that have already passed, and returns the current summary and remaining schedule.
        """
        user_id = self._get_verified_user_id(user_info)
        current_dt = datetime.now()

        playbook, items = await self.playbook_service.get_remaining_items(user_id, current_dt)

        if not playbook or not items:
            return json.dumps({
                "summary": "You have a completely free schedule for the rest of the day."
            })

        remaining_items = [
            {
                "time": item.event_time or "Anytime",
                "title": item.title,
                "category": item.category,
                "status": item.status,
            }
            for item in items
        ]

        return json.dumps({
            "daily_summary": playbook.morning_summary,
            "upcoming_events": remaining_items,
            "instruction": (
                "Respond warmly using the daily_summary and upcoming_events. "
                "Do not mention internal statuses or IDs."
            ),
        }, indent=2)

    async def respond_to_playbook_item(
            self,
            item_id: str,
            accepted: bool,
            user_info: Annotated[dict, InjectedState("userinfo")],
    ) -> str:
        """
        Use this tool when the user explicitly accepts or declines an optional activity or playbook prompt.
        When accepted=True, this tool AUTOMATICALLY schedules the reminder using the activity's scheduled time.
        Do NOT call schedule_reminder separately when using this tool.
        Args:
            item_id: The ID of the playbook item being responded to.
            accepted: True if the user wants to attend/participate, False if they decline.
        """
        user_id = self._get_verified_user_id(user_info)
        new_status = "ACCEPTED" if accepted else "DECLINED"
        now = datetime.now()

        # language=Cypher
        query = """
                MATCH (u:User {id: $user_id})-[:HAS_PLAYBOOK]->(:Playbook)-[:HAS_ITEM]->(pi:PlaybookItem {id: $item_id})
                SET pi.status = $new_status, pi.updated_at = datetime($updated_at)
                RETURN pi.title AS title,
                       pi.event_time AS event_time,
                       coalesce(pi.lead_minutes, 15) AS lead_minutes,
                       coalesce(pi.validity_minutes, 45) AS validity_minutes
                """
        res = await self.db_client.execute_query(query, {
            "user_id": user_id,
            "item_id": item_id,
            "new_status": new_status,
            "updated_at": now.isoformat(),
        })
        data = res.get("data", [])

        if not data:
            return json.dumps({"error": "Playbook item not found."})

        item = data[0]
        if accepted and item.get("event_time"):
            event_dt = PlaybookService.resolve_event_datetime(now.date(), str(item["event_time"]))
            if event_dt:
                lead_minutes = int(item.get("lead_minutes", 15))
                validity_minutes = int(item.get("validity_minutes", 45))
                due_dt = event_dt - timedelta(minutes=lead_minutes)

                if due_dt > now:
                    text = (
                        f"It is almost time for {item['title']}. "
                        f"It starts at {event_dt.strftime('%I:%M %p').lstrip('0')}."
                    )
                    await self.playbook_service.create_and_schedule_task(
                        user_id=user_id,
                        playbook_item_id=item_id,
                        text=text,
                        due_dt=due_dt,
                        validity_minutes=validity_minutes,
                        current_dt=now,
                    )
                    return json.dumps({
                        "success": True,
                        "message": f"Activity accepted. Reminder set for {due_dt.strftime('%I:%M %p').lstrip('0')}.",
                    })

        return json.dumps({"success": True, "message": f"Activity marked as {new_status}."})

    def get_tools(self) -> list:
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_remaining_daily_plans,
                name="get_remaining_daily_plans",
                description=self.get_remaining_daily_plans.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.respond_to_playbook_item,
                name="respond_to_playbook_item",
                description=self.respond_to_playbook_item.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
        ]