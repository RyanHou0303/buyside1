import pandas as pd
def to_utc(value)->pd.Timestamp:
    """
    convert to UTC time
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def first_open_time(
        available_at,
        data:pd.Dataframe,
        open_col:str = "market_open_at",
        
    )->pd.Timestamp: 
    """
    after the information is available, 
    what is the first open time

    avalable_at: when the information is available
    """
    timestamp = to_utc(available_at)
    opens = pd.DatetimeIndex(
        pd.to_datetime(data[open_col],utc=True).sort_values()
    )

    location = opens.searchsorted(timestamp,'right')
    if location >len(opens):
        raise IndexError("No future trading session available")
    return opens[location]

    

