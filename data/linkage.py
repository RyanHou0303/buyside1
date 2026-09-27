from __future__ import annotations
import pandas as pd

def resolve_event_to_listing(
        event:pd.Series,
):
    
    """
    events was like a serial of events
    happened in a serial of timestamp
    events:
    pd.Series(["merge", 2020-01-03], index=[name, decisions_at])

    """
    decision_at = pd.Timestamp(event["available_at"])
    event = decision_at.date()
