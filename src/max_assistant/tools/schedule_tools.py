# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines Pydantic models for the schedule-related tools and neo4j nodes
"""
import json
import asyncio
import logging
from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState

from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.models.schedule_models import (
    Appointment,
    DailyRoutine,
)
from max_assistant.tools.base_provider import BaseToolProvider


logger = logging.getLogger(__name__)


class ScheduleTools(BaseToolProvider):
    def __init__(self, db_client: Neo4jClient, llm: ChatOllama = None):
        """
        Initializes the toolset with a specific Neo4j client.
        """
        super().__init__(db_client, llm)
        logger.debug("PersonTools initialized with a Neo4j client.")

    async def get_appointments_for_date(
            self,
            target_date: str,
            user_info: Annotated[dict, InjectedState("userinfo")]
    ) -> str:
        """
        Use this tool if the user asks specifically for 'appointments'
        and NOT for their general 'schedule'. For general schedule
        questions, use 'get_full_schedule'.
        The target_date arg must be a valid iso date string.
        """
        logger.info(f"Tool: get_appointments_for_date for {target_date}")

        user_id = self._get_verified_user_id(user_info)


        query = """
                WITH datetime($targetDate) AS dt
                MATCH (u:User {id: $user_id})
                      -[:HAS_YEAR]->(:Year {year: dt.year})
                      -[:HAS_MONTH]->(:Month {month: dt.month})
                      -[:HAS_DAY]->(:Day {day: dt.day})
                      -[:HAS_APPOINTMENT]->(appt:Appointment)
                RETURN properties(appt) AS appointment  
                """

        params = {
            "targetDate": target_date,
            "user_id": user_id,
        }

        # Call the same helper with a different model and key
        return await self._query_and_validate_nodes(
            query=query,
            model_class=Appointment,
            result_key="appointment",
            params=params,
        )


    async def get_routines_for_date(
            self,
            target_date: str,
            user_info: Annotated[dict, InjectedState("userinfo")]
        ) -> str:
        """
        Use this tool if the user asks specifically for 'activities', 'daily routines',
        'meal times', or 'medication times'. For general schedule
        questions, use 'get_full_schedule'. The target_date arg must be a valid iso date string.
        """
        logger.info(f"Tool: get_routines_for_date for {target_date}")

        user_id = self._get_verified_user_id(user_info)

        query = """
                MATCH (u:User {id: $user_id})
                WITH datetime($targetDate) AS dt
                WITH CASE dt.dayOfWeek
                    WHEN 1 THEN 'Monday' WHEN 2 THEN 'Tuesday' WHEN 3 THEN 'Wednesday'
                    WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday' WHEN 6 THEN 'Saturday'
                    ELSE 'Sunday' 
                END AS dowString
                MATCH (u:User {id: $user_id})
                OPTIONAL MATCH (u)-[:ATTENDS]->(routine:DailyRoutine)
                WHERE dowString IN routine.dayOfWeek
                WITH routine WHERE routine IS NOT NULL
                RETURN properties(routine) AS routine 
                """

        params = {
            "targetDate": target_date,
            "user_id": user_id,
        }

        # Call the helper
        return await self._query_and_validate_nodes(
            query=query,
            model_class=DailyRoutine,
            result_key="routine",
            params=params,
        )

    async def get_full_schedule(
            self,
            target_date: str,
            user_info: Annotated[dict, InjectedState("userinfo")],
        ) -> str:
        """
        ONLY use this tool if the user specifically asks for their "full schedule",
        use the tool get_remaining_daily_plans for other schedule-related questions.
        The target_date arg must be a valid iso date string.
        """
        logger.info(f"Tool: get_full_schedule for {target_date}")

        try:
            # Run both tools concurrently, passing the same args object downstream
            results = await asyncio.gather(
                self.get_appointments_for_date(target_date, user_info),
                self.get_routines_for_date(target_date, user_info)
            )

            appts_json_str, routines_json_str = results

            # Parse the JSON results from both tools
            appts = json.loads(appts_json_str)
            routines = json.loads(routines_json_str)

            # Check if either tool returned an error
            if isinstance(appts, dict) and 'error' in appts:
                logger.error(f"Error from get_appointments: {appts}")
                return json.dumps(appts)
            if isinstance(routines, dict) and 'error' in routines:
                logger.error(f"Error from get_routines: {routines}")
                return json.dumps(routines)

            appts_list = appts if isinstance(appts, list) else []
            routines_list = routines if isinstance(routines, list) else []

            # Combine the two lists
            combined_list = []
            for item in appts_list + routines_list:
                combined_list.append({key: item.get(key) for key in ['time', 'title', 'duration', 'details']})

            # Sort the combined list by 'time'
            # We use .get() with a default to prevent errors if a key is missing
            sorted_list = sorted(
                combined_list,
                key=lambda x: x.get('time', '23:59:59')
            )

            logger.info(f"Returning combined schedule with {len(sorted_list)} items.")
            return json.dumps(sorted_list, indent=2)

        except Exception as e:
            # Catch-all for any other errors (like JSON parsing)
            error_type = e.__class__.__name__
            logger.error(f"Unexpected error in get_full_schedule: {e}", exc_info=True)
            return json.dumps({"error": f"Internal error combining schedule: {error_type}"})

    async def create_appointment(
            self,
            title: str,
            time: str,
            date: str,
            details: str | None,
            duration: int | None,
            user_info: Annotated[dict, InjectedState("userinfo")],
    ) -> str:
        """
        Creates a new appointment for the user.
        """
        logger.info(f"Tool: Calling create_appointment with args: {title}")

        user_id = self._get_verified_user_id(user_info)

        params = {
            "title": title,
            "time": time,
            "date": date,
            "details": details,
            "duration": duration,
            "user_id": user_id
        }

        query = """
            // 1. Fetch the User node first to anchor the transaction
            MATCH (u:User {id: $user_id})
            
            // 2. Instantiate your date variable cleanly as 'd' so it matches downstream properties
            WITH u, date($date) AS d
            
            // 3. MERGE the entire time tree branch securely relative to this specific User
            MERGE (u)-[:HAS_YEAR]->(y:Year {year: d.year})
            MERGE (y)-[:HAS_MONTH]->(m:Month {month: d.month, name: format(d, 'MMM'), year: d.year})
            MERGE (m)-[:HAS_DAY]->(day:Day {day: d.day, month: d.month, year: d.year})
            
            // 4. Create the new Appointment node inside the active transaction scope
            CREATE (a:Appointment {
                id: randomUUID(),
                title: $title,
                time: $time,
                date: $date,
                details: $details,
                duration: $duration
            })
            
            // 5. Securely link the unique timeline leaf node to the new appointment record
            MERGE (day)-[:HAS_APPOINTMENT]->(a)
                """

        return await self._safe_execute_query(query, params)

    async def get_activities_info(self, user_info: Annotated[dict, InjectedState("userinfo")]) -> str:
        """
        Use this tool to get details about available Activities.
        The rating is an indicator of how much the user dislikes or enjoys an activity, favorites should include
        ratings with values of 'enjoy' and 'ok' but exclude 'dislike'.  If responding with details about a disliked
        activity, the user's rating should be
        gently acknowledged, for example, "There is bingo at 7 pm, but I know you dislike it."
        Pay attention to the day of the week the activity is offered and the current date.
        Use this to answer questions about activities like 'What are my favorite activities?' or to suggest activities.
        """
        logger.info("Tool: get_routine_info for type")

        query = """
                MATCH (u:User {id: $user_id})
                OPTIONAL MATCH (u)-[:ATTENDS]->(routine:DailyRoutine)
                WHERE routine.type in $type_list
                WITH routine WHERE routine IS NOT NULL
                RETURN properties(routine) AS routine 
                """

        user_id = self._get_verified_user_id(user_info)

        params = {
            "type_list": ['activity', 'exercise'],
            "user_id": user_id
        }

        return await self._query_and_validate_nodes(
            query=query,
            model_class=DailyRoutine,
            result_key="routine",
            params=params,
        )

    def get_tools(self) -> list:
        """
        Returns a list of all tool methods bound to this instance.
        """
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_appointments_for_date,
                name="get_appointments_for_date",
                description=self.get_appointments_for_date.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_routines_for_date,
                name="get_routines_for_date",
                description=self.get_routines_for_date.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_full_schedule,
                name="get_full_schedule",
                description=self.get_full_schedule.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.create_appointment,
                name="create_appointment",
                description=self.create_appointment.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_activities_info,
                name="get_activities_info",
                description=self.get_activities_info.__doc__,
                handle_tool_error=self.format_system_tool_error,
            )
        ]