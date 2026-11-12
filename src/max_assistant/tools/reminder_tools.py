# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines LangGraph tools for scheduling context-aware reminders using task nodes and saving them to Neo4j.
Supports either relative delays or specific target times alongside validity windows and wake flags.
"""
from datetime import datetime, timedelta
import json
import logging
from typing import Annotated, Optional

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState

from max_assistant.tools.registry import BaseToolProvider
from max_assistant.clients.neo4j_client import Neo4jClientError, Neo4jCircuitBreakerError
from max_assistant.utils.decorators import requires_db

logger = logging.getLogger(__name__)


class ReminderTools(BaseToolProvider):
    """
    A class that encapsulates reminder and task scheduling tools
    backed by a Neo4j client instance.
    """

    def __init__(self, db_client, time_scheduler, llm=None):
        super().__init__(db_client, llm)
        self.time_scheduler = time_scheduler
        logger.debug("ReminderTools initialized with a Neo4j client and PersistentTimeScheduler.")

    @staticmethod
    def _parse_target_time(target_str: str, now: datetime) -> Optional[datetime]:
        """
        Attempts to parse a target time string in various common formats (e.g., '13:30', '1:30 PM', ISO).
        If the resolved time is in the past for today, it rolls over and becomes a time tomorrow.
        """
        clean_str = target_str.strip().upper()

        # Try ISO / full datetime formats first
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(clean_str, fmt)
            except ValueError:
                pass

        # Try time-only formats
        for fmt in ("%I:%M %p", "%I:%M%p", "%I %p", "%H:%M:%S", "%H:%M"):
            try:
                parsed_time = datetime.strptime(clean_str, fmt).time()
                candidate_dt = datetime.combine(now.date(), parsed_time)
                # If target time has already passed today, schedule for tomorrow
                if candidate_dt <= now:
                    candidate_dt += timedelta(days=1)
                return candidate_dt
            except ValueError:
                pass

        return None

    @requires_db
    async def schedule_reminder(
            self,
            message: str,
            validity_minutes: float,
            can_wake_user: bool,
            user_info: Annotated[dict, InjectedState("userinfo")],
            delay_minutes: Optional[float] = None,
            target_time_str: Optional[str] = None,
    ) -> str:
        """
        Use this tool to schedule a reminder. Either 'delay_minutes' OR 'target_time_str' must be provided.

        Args:
            message: The reminder instruction or text to speak to the user.
            validity_minutes: How many minutes after the due time the reminder remains valid before expiring.
            can_wake_user: True if this reminder should sound when the user is asleep; False to wait until they wake up.
            delay_minutes: Elapsed minutes from now (e.g., 10 for "in 10 minutes").
            target_time_str: An absolute time string (e.g., "1:30 PM", "13:30", "14:00").
        """
        logger.info(
            f"Tool: schedule_reminder - message='{message}', delay={delay_minutes}, "
            f"target_time='{target_time_str}', validity={validity_minutes}m, can_wake={can_wake_user}"
        )

        user_id = self._get_verified_user_id(user_info)
        now = datetime.now()

        # Resolve due datetime from either elapsed minutes or absolute time string
        if delay_minutes is not None:
            due_datetime = now + timedelta(minutes=delay_minutes)
        elif target_time_str:
            due_datetime = self._parse_target_time(target_time_str, now)
            if not due_datetime:
                return json.dumps({
                    "error": "Invalid_Time_Format",
                    "instruction": f"Could not parse the time '{target_time_str}'. Please provide a valid time like '1:30 PM' or '14:00'."
                })
        else:
            return json.dumps({
                "error": "Missing_Time_Parameter",
                "instruction": "Please provide either 'delay_minutes' or 'target_time_str' to schedule the reminder."
            })

        try:
            expires_datetime = due_datetime + timedelta(minutes=validity_minutes)
            due_iso = due_datetime.isoformat()
            expires_iso = expires_datetime.isoformat()

            query = """
                MATCH (u:User {id: $user_id}) 
                WITH u, datetime($due_time) AS d, datetime($expires_at) AS exp
                CREATE (t:Task {
                    id: randomUUID(),
                    text: $text,
                    due_time: d,
                    expires_at: exp,
                    can_wake_user: $can_wake_user,
                    type: 'REMINDER',
                    status: 'PENDING'
                })
                MERGE (u)-[:HAS_YEAR]->(y:Year {year: d.year})
                MERGE (y)-[:HAS_MONTH]->(m:Month {month: d.month, year: d.year})
                ON CREATE SET m.name = format(d, 'MMM') 
                MERGE (m)-[:HAS_DAY]->(day:Day {day: d.day, month: d.month, year: d.year})
                MERGE (day)-[:HAS_TASK]->(t)
                RETURN t.id AS task_id
            """
            params = {
                "text": message,
                "due_time": due_iso,
                "expires_at": expires_iso,
                "can_wake_user": can_wake_user,
                "user_id": user_id,
            }

            result = await self.db_client.execute_query(query, params)

            data = result.get("data", [])
            if not data:
                logger.error("Failed to schedule reminder: Could not match User ID in database.")
                return json.dumps({
                    "error": "User_Not_Found",
                    "instruction": "Could not attach the reminder because your user profile was not found in the database. Please inform the user."
                })

            task_id = data[0].get("task_id")

            # Register with the priority heap
            await self.time_scheduler.schedule(
                task_id=task_id,
                user_id=user_id,
                text=message,
                due_epoch=due_datetime.timestamp(),
                expires_epoch=expires_datetime.timestamp(),
                can_wake_user=can_wake_user
            )

            readable_time = due_datetime.strftime("%I:%M %p").lstrip("0")
            return json.dumps({
                "success": True,
                "task_id": task_id,
                "message": f"Successfully scheduled reminder for {readable_time}: '{message}'",
                "due_time": due_iso,
                "expires_at": expires_iso,
                "can_wake_user": can_wake_user
            }, indent=2)

        except Neo4jCircuitBreakerError as e:
            logger.warning(f"Circuit Breaker blocked reminder scheduling: {e}")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The system database is currently offline. The reminder could NOT be saved. Apologize and inform the user.",
                "details": str(e)
            })

        except Neo4jClientError as e:
            logger.error(f"Database error in schedule_reminder: {e}")
            return json.dumps({
                "error": "Database_Unavailable",
                "instruction": "The database connection failed abruptly. Apologize to the user and inform them that the reminder could not be saved right now.",
                "details": str(e)
            })

        except Exception as e:
            logger.error(f"Unexpected error in schedule_reminder tool: {e}")
            return json.dumps({"error": "Failed to schedule reminder", "details": str(e)})

    def get_tools(self) -> list:
        """Returns the unified reminder tool bound to this instance."""
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.schedule_reminder,
                name="schedule_reminder",
                description=self.schedule_reminder.__doc__,
                handle_tool_error=self.format_system_tool_error,
            )
        ]