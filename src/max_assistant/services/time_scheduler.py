# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
import asyncio
import heapq
import time
from datetime import datetime, timedelta
import logging

from max_assistant.clients.neo4j_client import Neo4jClientError, Neo4jCircuitBreakerError
from max_assistant.events.bus import TOPIC_PLAYBOOK_GENERATE

logger = logging.getLogger(__name__)

class PersistentTimeScheduler:
    """Heap-based priority timer that evaluates validity and context before firing."""

    def __init__(self, db_client, event_bus, playbook_hour: int = 5, playbook_minute: int = 30):
        self.db_client = db_client
        self.event_bus = event_bus
        self.playbook_hour = playbook_hour
        self.playbook_minute = playbook_minute
        # Heap item: (due_epoch, task_id, user_id, text, expires_epoch, can_wake_user)
        self._heap = []
        self._wake_event = asyncio.Event()
        self._cron_wake_event = asyncio.Event()
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._cron_task: asyncio.Task | None = None

    def wake(self) -> None:
        """Signals the scheduler loop to immediately re-evaluate deferred tasks."""
        self._wake_event.set()
        self._cron_wake_event.set()

    async def schedule(self, task_id: str, user_id: str, text: str, due_epoch: float, expires_epoch: float, can_wake_user: bool):
        heapq.heappush(self._heap, (due_epoch, task_id, user_id, text, expires_epoch, can_wake_user))
        self._wake_event.set()

    async def start(self):
        if self._running:
            return
        self._running = True
        await self._hydrate_from_neo4j()
        self._loop_task = asyncio.create_task(self._run_loop())
        self._cron_task = asyncio.create_task(self._daily_playbook_cron_loop())

    async def stop(self):
        self._running = False
        self._wake_event.set()
        self._cron_wake_event.set()
        if self._loop_task:
            self._loop_task.cancel()
        if self._cron_task:
            self._cron_task.cancel()

    async def _daily_playbook_cron_loop(self):
        """Fires `playbook.generate` once daily at the configured pre-wake time (default 05:30)."""
        last_generated_date: str | None = None
        while self._running:
            now = datetime.now()
            target = now.replace(
                hour=self.playbook_hour,
                minute=self.playbook_minute,
                second=0,
                microsecond=0,
            )
            if now >= target:
                if last_generated_date != now.date().isoformat():
                    last_generated_date = now.date().isoformat()
                    await self.event_bus.publish(
                        TOPIC_PLAYBOOK_GENERATE,
                        {"date": last_generated_date, "trigger": "daily_cron"},
                    )
                target = target + timedelta(days=1)

            sleep_seconds = max(1.0, (target - datetime.now()).total_seconds())
            try:
                await asyncio.sleep(sleep_seconds)
            except asyncio.CancelledError:
                break

    async def _hydrate_from_neo4j(self):
        try:
            # language=Cypher
            query = """
            MATCH (u:User)-[:HAS_YEAR]->()-[:HAS_MONTH]->()-[:HAS_DAY]->()-[:HAS_TASK]->(t:Task)
            WHERE t.status = 'PENDING' AND t.type = 'REMINDER'
            RETURN t.id AS id, t.text AS text, t.due_time AS due_time, 
                   t.expires_at AS expires_at, t.can_wake_user AS can_wake, u.id AS user_id
            """
            result = await self.db_client.execute_query(query, {})
            for record in result.get("data", []):
                due_dt = datetime.fromisoformat(str(record["due_time"]))
                exp_dt = datetime.fromisoformat(str(record["expires_at"])) if record.get("expires_at") else due_dt + timedelta(hours=2)
                can_wake = bool(record.get("can_wake", False))

                heapq.heappush(self._heap, (
                    due_dt.timestamp(),
                    record["id"],
                    record["user_id"],
                    record["text"],
                    exp_dt.timestamp(),
                    can_wake
                ))
            logger.info(f"TimeScheduler: Hydrated {len(self._heap)} pending tasks from Neo4j.")
        except (Neo4jCircuitBreakerError, Neo4jClientError) as e:
            logger.error(f"TimeScheduler: Failed to hydrate from Neo4j: {e}")

    async def _run_loop(self):
        while self._running:
            if not self._heap:
                self._wake_event.clear()
                await self._wake_event.wait()
                continue

            now = time.time()
            due_ts, task_id, user_id, text, exp_ts, can_wake = self._heap[0]

            if due_ts <= now:
                # 1. Check if the reminder has expired
                if now > exp_ts:
                    heapq.heappop(self._heap)
                    logger.info(f"Task [{task_id}] expired prior to delivery. Marking EXPIRED.")
                    await self._set_task_status(task_id, "EXPIRED")
                    continue

                # 2. Query user presence and sleep state
                user_state = await self._get_user_state(user_id)
                is_present = user_state.get("presence") == "present"
                is_asleep = user_state.get("sleep_state") == "asleep"

                # Condition: User must be present AND (awake OR allowed to be woken)
                can_deliver = is_present and (not is_asleep or can_wake)

                if can_deliver:
                    heapq.heappop(self._heap)
                    claimed = await self._set_task_status(task_id, "COMPLETED")
                    if claimed:
                        await self.event_bus.publish("task.due", {
                            "task_id": task_id,
                            "user_id": user_id,
                            "text": text
                        })
                    else:
                        logger.debug(
                            f"Task [{task_id}] no longer PENDING in Neo4j (deleted or superseded). Skipping delivery."
                        )
                else:
                    # User is away or asleep: leave task in the heap, retry in 30s
                    logger.debug(f"Task [{task_id}] deferred: presence={is_present}, asleep={is_asleep}, can_wake={can_wake}")
                    self._wake_event.clear()
                    try:
                        # Wait 30 seconds or until woken by a presence/state transition
                        await asyncio.wait_for(self._wake_event.wait(), timeout=30.0)
                    except asyncio.TimeoutError:
                        pass
            else:
                sleep_delay = due_ts - now
                self._wake_event.clear()
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=sleep_delay)
                except asyncio.TimeoutError:
                    pass

    async def _get_user_state(self, user_id: str) -> dict:
        # language=Cypher
        query = """
        MATCH (u:User {id: $user_id})
        RETURN coalesce(u.presence, 'away') AS presence,
               coalesce(u.sleep_state, 'awake') AS sleep_state
        """
        res = await self.db_client.execute_query(query, {"user_id": user_id})
        data = res.get("data", [])
        return data[0] if data else {"presence": "present", "sleep_state": "awake"}

    async def _set_task_status(self, task_id: str, status: str) -> bool:
        # language=Cypher
        query = """
        MATCH (t:Task {id: $task_id, status: 'PENDING'})
        SET t.status = $status
        RETURN t.id AS id
        """
        res = await self.db_client.execute_query(query, {"task_id": task_id, "status": status})
        return bool(res.get("data"))