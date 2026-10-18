from .state import State


def transition(state: State, decision: dict, exogenous: dict) -> State:
    """
    S^M: the transition function.

    Given the current state, the decision made, and the realized exogenous
    information, return the next state. This is the "physics" of the system.

    Parameters
    ----------
    state : State
        Current state S_t.
    decision : dict
        Decision x_t produced by the policy. Expected keys:
          - 'bays_open' (int): number of bays to have open next period
    exogenous : dict
        Realized W_{t+1}. Expected keys:
          - 'arrivals' (int): cars that arrived this period
          - 'completions' (int): cars that finished service this period
          - 'wait_added' (float): total wait time accumulated this period

    Returns
    -------
    State
        Updated state S_{t+1}.
    """
    import copy
    next_state = copy.copy(state)

    next_state.bays_open = decision.get("bays_open", state.bays_open)

    arrivals = exogenous.get("arrivals", 0)
    completions = exogenous.get("completions", 0)
    wait_added = exogenous.get("wait_added", 0.0)

    # Cars still being washed stay in their bay, even if that bay is closing
    # (a closing bay takes no new cars but finishes its current one).
    in_service = state.bays_busy - completions
    waiting = state.queue_length + arrivals
    free_bays = max(0, next_state.bays_open - in_service)
    starts = min(free_bays, waiting)

    next_state.bays_busy = in_service + starts
    next_state.queue_length = waiting - starts
    next_state.cars_served += completions
    next_state.total_wait += wait_added

    return next_state
