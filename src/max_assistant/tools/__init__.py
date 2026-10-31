from .person_tools import PersonTools
from .schedule_tools import ScheduleTools
from .gmail_tools import GmailTools
from .general_query_tools import GeneralQueryTools
from .reminder_tools import ReminderTools

# A central list of all tool provider classes to be registered.
# To add a new tool, simply import it and add its class to this list.
ALL_TOOL_PROVIDERS = [
    PersonTools,
    ScheduleTools,
    GmailTools,
    # GeneralQueryTools,
    ReminderTools,
]

__all__ = [
    "ALL_TOOL_PROVIDERS",
    "PersonTools",
    "ScheduleTools",
    "GmailTools",
    # "GeneralQueryTools",
    "ReminderTools",
]