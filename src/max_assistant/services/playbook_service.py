# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Synthesizes a Daily Playbook each morning using Neo4j context (appointments, routines,
family messages/news) and Care Admin guidelines, scheduling timed reminders in PersistentTimeScheduler
and storing paced engagement items in Neo4j.
"""
from datetime import datetime, timedelta
import json
import logging
import re
from typing import Any, Dict, List, Optional
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from max_assistant.events.bus import (
    LocalEventBus,
    TOPIC_PLAYBOOK_GENERATE,
    TOPIC_PLAYBOOK_GENERATED,
)
from max_assistant.models.playbook_models import (
    CareAdminConfig,
    DailyPlaybook,
    PlaybookItem,
)

logger = logging.getLogger(__name__)


class PlaybookService:
    """Generates and persists daily playbooks and registers required reminders."""

    def __init__(
            self,
            db_client,
            llm: ChatOllama,
            time_scheduler,
            event_bus: LocalEventBus,
    ):
        self.db_client = db_client
        self.llm = llm
        self.time_scheduler = time_scheduler
        self.event_bus = event_bus

        self.event_bus.subscribe(TOPIC_PLAYBOOK_GENERATE, self.on_playbook_generate)

    async def on_playbook_generate(self, _topic: str, data: dict) -> None:
        """EventBus handler for `playbook.generate`."""
        target_date_str = data.get("date")
        # Explicit string cast fixes the 'Any | None' type warning
        current_dt = datetime.fromisoformat(str(target_date_str)) if target_date_str else datetime.now()
        user_id = data.get("user_id")

        if user_id:
            await self.generate_daily_playbook(user_id=str(user_id), current_dt=current_dt)
        else:
            await self.generate_for_all_users(current_dt=current_dt)

    async def generate_for_all_users(self, current_dt: Optional[datetime] = None) -> List[DailyPlaybook]:
        """Generates daily playbooks for all :User nodes in Neo4j."""
        current_dt = current_dt or datetime.now()
        res = await self.db_client.execute_query("MATCH (u:User) RETURN u.id AS id", {})
        playbooks: List[DailyPlaybook] = []

        for record in res.get("data", []):
            uid = record.get("id")
            if uid:
                pb = await self.generate_daily_playbook(user_id=str(uid), current_dt=current_dt)
                if pb:
                    playbooks.append(pb)

        return playbooks

    async def get_or_create_playbook(self, user_id: str, current_dt: Optional[datetime] = None) -> Optional[
        DailyPlaybook]:
        """Fetches today's existing Playbook from Neo4j, or synthesizes one if missing."""
        current_dt = current_dt or datetime.now()
        date_str = current_dt.date().isoformat()
        existing = await self.get_playbook(user_id=user_id, target_date=date_str)
        if existing is not None:
            return existing
        return await self.generate_daily_playbook(user_id=user_id, current_dt=current_dt)

    async def get_playbook(self, user_id: str, target_date: str) -> Optional[DailyPlaybook]:
        """Loads an existing Playbook and its PlaybookItems from Neo4j."""
        # language=Cypher
        query = """
                MATCH (u:User {id: $user_id})-[:HAS_PLAYBOOK]->(p:Playbook {date: $date})
                OPTIONAL MATCH (p)-[:HAS_ITEM]->(item:PlaybookItem)
                RETURN p.id AS id, p.date AS date, p.morning_summary AS morning_summary,
                collect(properties(item)) AS items LIMIT 1 \
                """
        res = await self.db_client.execute_query(query, {"user_id": user_id, "date": target_date})
        data = res.get("data", [])
        if not data:
            return None

        row = data[0]
        raw_items = [i for i in row.get("items", []) if i and i.get("id")]
        items = [PlaybookItem.model_validate(i) for i in raw_items]
        items.sort(key=lambda x: (-x.priority, x.event_time or "23:59"))

        return DailyPlaybook(
            id=str(row["id"]),
            user_id=user_id,
            date=str(row["date"]),
            morning_summary=str(row.get("morning_summary") or ""),
            items=items,
        )

    async def generate_daily_playbook(self, user_id: str, current_dt: Optional[datetime] = None) -> Optional[
        DailyPlaybook]:
        """Builds a structured DailyPlaybook trimmed to the current time, persists it, and schedules reminders."""
        current_dt = current_dt or datetime.now()
        date_str = current_dt.date().isoformat()
        time_str = current_dt.time().isoformat()

        logger.info(f"PlaybookService: Generating playbook for user='{user_id}' at {date_str} {time_str}...")

        care_cfg = await self._load_care_config(user_id)

        # Pass the full current_dt to Python-side filtering to keep DB queries time-agnostic
        context = await self._gather_daily_context(user_id, date_str, current_dt)

        playbook = await self._synthesize_with_llm(user_id, date_str, care_cfg, context)

        # Idempotent save: wipes out previous items and pending tasks for today
        await self._persist_playbook(playbook, current_dt)

        # Re-schedules only the upcoming trimmed reminders
        await self._schedule_required_reminders(playbook, care_cfg, current_dt)

        await self.event_bus.publish(
            TOPIC_PLAYBOOK_GENERATED,
            {
                "user_id": user_id,
                "playbook_id": playbook.id,
                "date": date_str,
                "item_count": len(playbook.items),
            },
        )
        logger.info(f"PlaybookService: Playbook '{playbook.id}' created with {len(playbook.items)} items.")
        return playbook

    async def _load_care_config(self, user_id: str) -> CareAdminConfig:
        """Loads CareProfile node or User properties; falls back to sensible defaults."""
        # language=Cypher
        query = """
                MATCH (u:User {id: $user_id})
                OPTIONAL MATCH (u)-[:HAS_CARE_PROFILE]->(cp:CareProfile)
                RETURN properties(cp) AS care_profile LIMIT 1 \
                """
        res = await self.db_client.execute_query(query, {"user_id": user_id})
        data = res.get("data", [])
        if data and data[0].get("care_profile"):
            return CareAdminConfig.model_validate(data[0]["care_profile"])
        return CareAdminConfig()

    async def _gather_daily_context(self, user_id: str, target_date: str, current_dt: datetime) -> Dict[str, Any]:
        """Queries Neo4j for today's context, filtering out passed events in Python for testability."""
        # 1. Appointments (No time logic in Cypher)
        # language=Cypher
        q_appts = """
                  WITH date($target_date) AS dt
                  MATCH (u:User {id: $user_id})-[:HAS_YEAR]->(:Year {year: dt.year})-[:HAS_MONTH]->(:Month {month: dt.month})-[:HAS_DAY]->(:Day {day: dt.day})-[:HAS_APPOINTMENT]->(appt:Appointment)
                  RETURN appt.id AS id, appt.title AS title, toString(appt.time) AS time,
                  appt.duration AS duration, appt.details AS details \
                  """

        # 2. Daily Routines (No time logic in Cypher)
        # language=Cypher
        q_routines = """
                     WITH date($target_date) AS dt
                     WITH CASE dt.dayOfWeek
                       WHEN 1 THEN 'Monday' 
                       WHEN 2 THEN 'Tuesday' 
                       WHEN 3 THEN 'Wednesday'
                       WHEN 4 THEN 'Thursday' 
                       WHEN 5 THEN 'Friday' 
                       WHEN 6 THEN 'Saturday'
                       ELSE 'Sunday'
                     END AS dow
                     MATCH (u:User {id: $user_id})-[:ATTENDS]->(r:DailyRoutine)
                     WHERE dow IN r.dayOfWeek
                     AND (r.startDate IS NULL OR r.startDate <= date($target_date))
                     AND (r.endDate IS NULL OR r.endDate >= date($target_date))
                     RETURN r.id AS id, r.title AS title, r.type AS type, toString(r.time) AS time,
                     r.duration AS duration, r.room AS room, r.details AS details, r.rating AS rating \
                     """

        # 3. Unplayed Family Messages
        # language=Cypher
        q_messages = """
                     MATCH (u:User {id: $user_id})-[:HAS_MESSAGE]->(m:FamilyMessage)
                     WHERE coalesce(m.status, 'UNREAD') = 'UNREAD'
                     RETURN m.id AS id, m.sender AS sender, m.text AS text, toString(m.created_at) AS created_at LIMIT 5 \
                     """

        # 4. Recent Family News
        # language=Cypher
        q_news = """
                 MATCH (u:User {id: $user_id})-[:HAS_NEWS]->(n:FamilyNews)
                 WHERE coalesce(n.shared_count, 0) < 2
                 RETURN n.id AS id, n.headline AS headline, n.details AS details LIMIT 5 \
                 """

        params = {"user_id": user_id, "target_date": target_date}
        appts_res = await self.db_client.execute_query(q_appts, params)
        routines_res = await self.db_client.execute_query(q_routines, params)
        msgs_res = await self.db_client.execute_query(q_messages, params)
        news_res = await self.db_client.execute_query(q_news, params)

        # ---------------------------------------------------------
        # Filter out passed events in Python (Testable & Deterministic)
        # ---------------------------------------------------------
        def is_upcoming(time_str: Optional[str]) -> bool:
            if not time_str:
                return True
            event_datetime = self._resolve_event_datetime(current_dt.date(), time_str)
            return event_datetime is None or event_datetime >= current_dt

        upcoming_appts = [a for a in appts_res.get("data", []) if is_upcoming(a.get("time"))]
        upcoming_routines = [r for r in routines_res.get("data", []) if is_upcoming(r.get("time"))]

        return {
            "appointments": upcoming_appts,
            "routines": upcoming_routines,
            "family_messages": msgs_res.get("data", []),
            "family_news": news_res.get("data", []),
        }

    async def _synthesize_with_llm(
            self,
            user_id: str,
            target_date: str,
            care_cfg: CareAdminConfig,
            context: Dict[str, Any],
    ) -> DailyPlaybook:
        """Uses the LLM to produce a dementia-friendly playbook, with deterministic fallback."""
        system_prompt = (
            "You are a dementia care planning assistant creating a calm, structured Daily Playbook.\n"
            "Rules:\n"
            "1. Categorize every event into one of: REQUIRED, MEAL, MEDICATION, OPTIONAL_ACTIVITY, FAMILY_MESSAGE, FAMILY_NEWS.\n"
            "2. Appointments and medical visits are REQUIRED with lead_minutes="
            f"{care_cfg.appointment_lead_minutes}.\n"
            "3. Meals are MEAL with lead_minutes="
            f"{care_cfg.meal_lead_minutes}.\n"
            "4. Social activities (like Bingo, crafts, music) are OPTIONAL_ACTIVITY unless marked mandatory; "
            "do not include activities rated 'dislike'.\n"
            "5. Keep summaries warm, clear, and under 2 short sentences suitable for someone with mild-to-moderate dementia.\n"
            f"6. Care Admin notes: {care_cfg.categorization_notes or 'None'}\n"
            "Return ONLY valid JSON matching:\n"
            '{"morning_summary": "...", "items": [{"category": "...", "title": "...", "summary": "...", '
            '"event_time": "HH:MM or null", "lead_minutes": 15, "validity_minutes": 60, "priority": 5, '
            '"sender": null, "room": null}]}'
        )
        user_prompt = json.dumps({"date": target_date, "context": context}, default=str)

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            raw_text = str(response.content).strip()
            json_match = re.search(r"\{.*}", raw_text, re.DOTALL)
            parsed = json.loads(json_match.group(0) if json_match else raw_text)

            items: List[PlaybookItem] = []
            for raw_item in parsed.get("items", []):
                items.append(
                    PlaybookItem(
                        id=str(uuid.uuid4()),
                        category=raw_item.get("category", "OPTIONAL_ACTIVITY"),
                        title=str(raw_item.get("title", "Activity")),
                        summary=str(raw_item.get("summary", raw_item.get("title", ""))),
                        event_time=raw_item.get("event_time"),
                        lead_minutes=int(raw_item.get("lead_minutes", 0)),
                        validity_minutes=int(raw_item.get("validity_minutes", 60)),
                        priority=int(raw_item.get("priority", 5)),
                        status="QUEUED",
                        sender=raw_item.get("sender"),
                        room=raw_item.get("room"),
                    )
                )

            return DailyPlaybook(
                id=f"pb-{user_id}-{target_date}",
                user_id=user_id,
                date=target_date,
                morning_summary=str(parsed.get("morning_summary") or self._build_fallback_summary(target_date, items)),
                items=items,
            )
        except Exception as e:
            logger.warning(f"PlaybookService: LLM synthesis failed ({e}); using deterministic fallback.")
            return self._deterministic_fallback(user_id, target_date, care_cfg, context)

    def _deterministic_fallback(
            self,
            user_id: str,
            target_date: str,
            care_cfg: CareAdminConfig,
            context: Dict[str, Any],
    ) -> DailyPlaybook:
        """Rule-based fallback generator if LLM output is unavailable or malformed."""
        items: List[PlaybookItem] = []

        for appt in context.get("appointments", []):
            title = str(appt.get("title", "Appointment"))
            items.append(
                PlaybookItem(
                    id=str(uuid.uuid4()),
                    category="REQUIRED",
                    title=title,
                    summary=f"You have {title} today at {appt.get('time', '')}.",
                    event_time=str(appt.get("time")) if appt.get("time") else None,
                    lead_minutes=care_cfg.appointment_lead_minutes,
                    validity_minutes=60,
                    priority=10,
                )
            )

        for r in context.get("routines", []):
            if str(r.get("rating", "")).lower() == "dislike":
                continue
            title = str(r.get("title", "Routine"))
            r_type = str(r.get("type", "")).lower()
            is_meal = r_type == "meal" or any(k in title.lower() for k in care_cfg.meal_keywords)
            is_med = r_type == "medication" or "med" in title.lower()

            if is_med:
                cat, lead, prio = "MEDICATION", 5, 10
            elif is_meal:
                cat, lead, prio = "MEAL", care_cfg.meal_lead_minutes, 8
            else:
                cat, lead, prio = "OPTIONAL_ACTIVITY", care_cfg.activity_ask_ahead_minutes, 5

            items.append(
                PlaybookItem(
                    id=str(uuid.uuid4()),
                    category=cat,  # type: ignore
                    title=title,
                    summary=f"{title} is scheduled at {r.get('time', '')}.",
                    event_time=str(r.get("time")) if r.get("time") else None,
                    lead_minutes=lead,
                    validity_minutes=45,
                    priority=prio,
                    room=r.get("room"),
                )
            )

        for msg in context.get("family_messages", []):
            sender = str(msg.get("sender", "Family"))
            items.append(
                PlaybookItem(
                    id=str(uuid.uuid4()),
                    category="FAMILY_MESSAGE",
                    title=f"Message from {sender}",
                    summary=str(msg.get("text", "")),
                    priority=4,
                    sender=sender,
                )
            )

        for news in context.get("family_news", []):
            items.append(
                PlaybookItem(
                    id=str(uuid.uuid4()),
                    category="FAMILY_NEWS",
                    title=str(news.get("headline", "Family Update")),
                    summary=str(news.get("details", news.get("headline", ""))),
                    priority=3,
                )
            )

        return DailyPlaybook(
            id=f"pb-{user_id}-{target_date}",
            user_id=user_id,
            date=target_date,
            morning_summary=self._build_fallback_summary(target_date, items),
            items=items,
        )

    @staticmethod
    def _build_fallback_summary(target_date: str, items: List[PlaybookItem]) -> str:
        dt = datetime.fromisoformat(target_date)
        day_name = dt.strftime("%A, %B %d")
        required = [i for i in items if i.category in ("REQUIRED", "MEDICATION")]
        if required:
            first = required[0]
            return f"Good morning! Today is {day_name}. Your main event today is {first.title} at {first.event_time or 'later today'}."
        return f"Good morning! Today is {day_name}. You have a relaxed schedule ahead today."

    async def _persist_playbook(self, playbook: DailyPlaybook, current_dt: datetime) -> None:
        """Persists the Playbook using a Python-provided timestamp and deletes old pending tasks."""
        # language=Cypher
        query = """
                MATCH (u:User {id: $user_id})
                MERGE (p:Playbook {id: $playbook_id})
                SET p.date = $date,
                p.morning_summary = $morning_summary,
                p.updated_at = datetime($updated_at)
                MERGE (u)-[:HAS_PLAYBOOK]->(p)
                WITH p

                // Wipe old playbook items and their spawned tasks to prevent duplicate alarms
                OPTIONAL MATCH (p)-[:HAS_ITEM]->(old_item:PlaybookItem)
                OPTIONAL MATCH (old_item)-[:SPAWNED_TASK]->(old_task:Task)
                DETACH \
                DELETE old_item, old_task
                WITH p

                // Insert new trimmed schedule
                UNWIND $items AS item
                CREATE ( \
                  pi:PlaybookItem {
                    id: item.id,
                    category: item.category,
                    title: item.title,
                    summary: item.summary,
                    event_time: item.event_time,
                    lead_minutes: item.lead_minutes,
                    validity_minutes: item.validity_minutes,
                    priority: item.priority,
                    status: item.status,
                    sender: item.sender,
                    room: item.room,
                    updated_at: datetime($updated_at)
                  } \
                )
                CREATE (p)-[:HAS_ITEM]->(pi) \
                """
        await self.db_client.execute_query(
            query,
            {
                "user_id": playbook.user_id,
                "playbook_id": playbook.id,
                "date": playbook.date,
                "morning_summary": playbook.morning_summary,
                "updated_at": current_dt.isoformat(),
                "items": [item.model_dump() for item in playbook.items],
            },
        )

    async def _schedule_required_reminders(self, playbook: DailyPlaybook, care_cfg: CareAdminConfig,
                                           current_dt: datetime) -> None:
        """Creates Neo4j REMINDER tasks and registers them with PersistentTimeScheduler."""
        base_date = datetime.fromisoformat(playbook.date).date()

        for item in playbook.items:
            if item.category not in ("REQUIRED", "MEDICATION", "MEAL") or not item.event_time:
                continue

            event_dt = self._resolve_event_datetime(base_date, item.event_time)
            if event_dt is None:
                continue

            # Schedule prep reminder (e.g., 45 min before appointment)
            prep_dt = event_dt - timedelta(minutes=item.lead_minutes)
            if prep_dt > current_dt:
                prep_text = (
                    f"You have {item.title} at {event_dt.strftime('%I:%M %p').lstrip('0')}. "
                    f"It is time to start getting ready."
                )
                await self.create_and_schedule_task(
                    user_id=playbook.user_id,
                    playbook_item_id=item.id,
                    text=prep_text,
                    due_dt=prep_dt,
                    validity_minutes=max(item.lead_minutes, 15),
                    current_dt=current_dt
                )

            # For REQUIRED appointments, also schedule a departure/pickup reminder
            if item.category == "REQUIRED" and care_cfg.departure_lead_minutes > 0:
                depart_dt = event_dt - timedelta(minutes=care_cfg.departure_lead_minutes)
                if depart_dt > current_dt:
                    depart_text = (
                        f"Your {item.title} is in {care_cfg.departure_lead_minutes} minutes "
                        f"at {event_dt.strftime('%I:%M %p').lstrip('0')}."
                    )
                    await self.create_and_schedule_task(
                        user_id=playbook.user_id,
                        playbook_item_id=item.id,
                        text=depart_text,
                        due_dt=depart_dt,
                        validity_minutes=30,
                        current_dt=current_dt
                    )

            # Mark playbook item as SCHEDULED
            await self.db_client.execute_query(
                "MATCH (pi:PlaybookItem {id: $id}) SET pi.status = 'SCHEDULED'",
                {"id": item.id},
            )

    async def create_and_schedule_task(
            self,
            user_id: str,
            playbook_item_id: str,
            text: str,
            due_dt: datetime,
            validity_minutes: int,
            current_dt: datetime,
    ) -> None:
        expires_dt = due_dt + timedelta(minutes=validity_minutes)
        # language=Cypher
        query = """
                MATCH (u:User {id: $user_id})
                MATCH (pi:PlaybookItem {id: $item_id})
                WITH u, pi, datetime($due_time) AS d, datetime($expires_at) AS exp, datetime($created_at) AS cat
                CREATE ( \
                  t:Task {
                    id: randomUUID(),
                    text: $text,
                    due_time: d,
                    expires_at: exp,
                    can_wake_user: false,
                    type: 'REMINDER',
                    status: 'PENDING',
                    source: 'PLAYBOOK',
                    created_at: cat
                  } \
                )
                MERGE (u)-[:HAS_YEAR]->(y:Year {year: d.year})
                MERGE (y)-[:HAS_MONTH]->(m:Month {month: d.month, year: d.year})
                  ON CREATE SET m.name = format(d, 'MMM')
                MERGE (m)-[:HAS_DAY]->(day:Day {day: d.day, month: d.month, year: d.year})
                MERGE (day)-[:HAS_TASK]->(t)
                MERGE (pi)-[:SPAWNED_TASK]->(t)
                RETURN t.id AS task_id \
                """
        res = await self.db_client.execute_query(
            query,
            {
                "user_id": user_id,
                "item_id": playbook_item_id,
                "text": text,
                "due_time": due_dt.isoformat(),
                "expires_at": expires_dt.isoformat(),
                "created_at": current_dt.isoformat(),
            },
        )
        data = res.get("data", [])
        if data and self.time_scheduler is not None:
            await self.time_scheduler.schedule(
                task_id=data[0]["task_id"],
                user_id=user_id,
                text=text,
                due_epoch=due_dt.timestamp(),
                expires_epoch=expires_dt.timestamp(),
                can_wake_user=False,
            )

    @staticmethod
    def _resolve_event_datetime(base_date, time_str: str) -> Optional[datetime]:
        clean = time_str.strip()
        for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M%p"):
            try:
                parsed_t = datetime.strptime(clean.upper(), fmt).time()
                return datetime.combine(base_date, parsed_t)
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(clean)
        except ValueError:
            return None