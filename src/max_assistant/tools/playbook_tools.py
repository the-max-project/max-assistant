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

from max_assistant.tools.base_provider import BaseToolProvider

logger = logging.getLogger(__name__)


class PlaybookTools(BaseToolProvider):
    def __init__(self, db_client, playbook_service, llm=None):
        super().__init__(db_client, llm)
        self.playbook_service = playbook_service

    async def get_remaining_daily_plans(
            self,
            user_info: Annotated[dict, InjectedState("userinfo")],
    ) -> str:
        """
        Use this tool when the user asks about their schedule, plans, or what is happening today.
        It regenerates the daily playbook, automatically trimming events that have already passed,
        and returns the current summary and schedule.
        """
        user_id = self._get_verified_user_id(user_info)
        current_dt = datetime.now()

        # Calling the generator multiple times is safe; it overwrites the day's playbook idempotently.
        playbook = await self.playbook_service.generate_daily_playbook(user_id, current_dt)

        if not playbook or not playbook.items:
            return json.dumps({
                "summary": "You have a completely free schedule for the rest of the day."
            })

        remaining_items = [
            {"time": item.event_time or "Anytime", "title": item.title, "category": item.category}
            for item in playbook.items
        ]
        remaining_items.sort(key=lambda x: x["time"] if x["time"] != "Anytime" else "23:59")

        return json.dumps({
            "daily_summary": playbook.morning_summary,
            "upcoming_events": remaining_items,
            "instruction": "Respond warmly using the daily_summary. You may mention the upcoming_events."
        }, indent=2)

    async def respond_to_playbook_item(
            self,
            item_id: str,
            accepted: bool,
            user_info: Annotated[dict, InjectedState("userinfo")],
    ) -> str:
        """
        Use this tool when the user explicitly accepts or declines an optional activity or playbook prompt.
        Args:
            item_id: The ID of the playbook item being responded to.
            accepted: True if the user wants to attend/participate, False if they decline.
        """
        user_id = self._get_verified_user_id(user_info)
        new_status = "ACCEPTED" if accepted else "DECLINED"
        now = datetime.now()

        # language=Cypher
        query = """
                MATCH (pi:PlaybookItem {id: $item_id})
                SET pi.status = $new_status, pi.updated_at = datetime($updated_at)
                RETURN pi.title AS title, pi.event_time AS event_time, pi.lead_minutes AS lead_minutes \
                """
        res = await self.db_client.execute_query(query, {
            "item_id": item_id,
            "new_status": new_status,
            "updated_at": now.isoformat()
        })
        data = res.get("data", [])

        if not data:
            return json.dumps({"error": "Playbook item not found."})

        item = data[0]
        if accepted and item.get("event_time"):
            base_date = now.date()
            time_str = item["event_time"]
            event_dt = None

            for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M%p"):
                try:
                    parsed_t = datetime.strptime(time_str.upper().strip(), fmt).time()
                    event_dt = datetime.combine(base_date, parsed_t)
                    break
                except ValueError:
                    pass

            if event_dt:
                lead_minutes = item.get("lead_minutes", 15)
                due_dt = event_dt - timedelta(minutes=lead_minutes)

                if due_dt > now:
                    text = f"It is almost time for {item['title']}. It starts at {event_dt.strftime('%I:%M %p').lstrip('0')}."

                    # Delegate the task creation and scheduling back to the playbook service
                    await self.playbook_service.create_and_schedule_task(
                        user_id=user_id,
                        playbook_item_id=item_id,
                        text=text,
                        due_dt=due_dt,
                        validity_minutes=45,
                        current_dt=now
                    )
                    return json.dumps({"success": True,
                                       "message": f"Activity accepted. Reminder set for {due_dt.strftime('%I:%M %p')}."})

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
            )
        ]