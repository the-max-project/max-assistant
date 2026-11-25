# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Pydantic models for Care Admin guidelines and the LLM-generated Daily Playbook.
"""
from typing import List, Literal, Optional
from pydantic import Field
from max_assistant.models.base import BaseNeo4jModel

PlaybookCategory = Literal[
    "REQUIRED",
    "MEAL",
    "MEDICATION",
    "OPTIONAL_ACTIVITY",
    "FAMILY_MESSAGE",
    "FAMILY_NEWS",
]

PlaybookItemStatus = Literal[
    "QUEUED",
    "SCHEDULED",
    "PROMPTED",
    "ACCEPTED",
    "DECLINED",
    "COMPLETED",
    "EXPIRED",
]


class CareAdminConfig(BaseNeo4jModel):
    """Care-admin configuration governing event categorization and dementia pacing."""
    appointment_lead_minutes: int = 45
    departure_lead_minutes: int = 15
    meal_lead_minutes: int = 15
    activity_ask_ahead_minutes: int = 15
    max_items_per_turn: int = 2
    prompt_cooldown_minutes: int = 20
    wakeup_briefing_hour_start: int = 5
    wakeup_briefing_hour_end: int = 11
    required_keywords: List[str] = Field(
        default_factory=lambda: ["doctor", "appointment", "physio", "dentist", "nurse", "medication"]
    )
    meal_keywords: List[str] = Field(
        default_factory=lambda: ["breakfast", "lunch", "dinner", "supper", "meal"]
    )
    categorization_notes: Optional[str] = None


class PlaybookItem(BaseNeo4jModel):
    """A single actionable, scheduled, or conversational item in a user's Daily Playbook."""
    id: str
    category: PlaybookCategory
    title: str
    summary: str
    event_time: Optional[str] = None        # HH:MM or ISO timestamp
    ask_by_time: Optional[str] = None       # Latest time to ask about optional activity
    lead_minutes: int = 0
    validity_minutes: int = 60
    priority: int = 5                       # 1 (low) to 10 (urgent)
    status: PlaybookItemStatus = "QUEUED"
    sender: Optional[str] = None
    media_url: Optional[str] = None
    room: Optional[str] = None


class DailyPlaybook(BaseNeo4jModel):
    """Represents a synthesized daily engagement plan for a resident."""
    id: str
    user_id: str
    date: str                               # YYYY-MM-DD
    morning_summary: str
    items: List[PlaybookItem] = Field(default_factory=list)