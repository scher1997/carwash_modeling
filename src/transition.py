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

    next_state.queue_length = max(0, state.queue_length + arrivals - completions)
    next_state.bays_busy = min(next_state.bays_open, next_state.queue_length + completions - arrivals + state.bays_busy)
    next_state.cars_served += completions
    next_state.total_wait += wait_added

    return next_state
