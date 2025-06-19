import datetime
import json
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "agent_logs.jsonl")

def log_agent_event(event: dict, user_id: str = None):
    """Append an agent event to the local JSONL log file, with optional user_id."""
    event["timestamp"] = datetime.datetime.utcnow().isoformat()
    if user_id:
        event["user_id"] = user_id
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")