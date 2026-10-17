from state import State


def policy_always_open(state: State, params: dict) -> dict:
    """
    Baseline policy: keep a fixed number of bays open at all times.
    params: {'bays': int}
    """
    return {"bays_open": params.get("bays", 1)}


def policy_threshold(state: State, params: dict) -> dict:
    """
    Open an extra bay when queue length exceeds a threshold; close it when
    the queue drops below a lower threshold.
    params: {'open_threshold': int, 'close_threshold': int, 'max_bays': int}
    """
    open_thresh = params.get("open_threshold", 5)
    close_thresh = params.get("close_threshold", 2)
    max_bays = params.get("max_bays", 3)

    if state.queue_length >= open_thresh:
        bays = min(state.bays_open + 1, max_bays)
    elif state.queue_length <= close_thresh:
        bays = max(state.bays_open - 1, 1)
    else:
        bays = state.bays_open

    return {"bays_open": bays}


# Register policies here for easy lookup in simulate.py
POLICIES = {
    "always_open": policy_always_open,
    "threshold": policy_threshold,
}
