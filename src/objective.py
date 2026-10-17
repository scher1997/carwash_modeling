from .state import State


def cost(state: State, decision: dict, params: dict) -> float:
    """
    C_t: contribution function evaluated at each time step.

    Returns the cost (or negative reward) incurred at this step.
    Lower is better. Adjust weights in params to change the trade-off.

    Parameters
    ----------
    state : State
        Current state S_t (before transition).
    decision : dict
        Decision x_t.
    params : dict
        Weights and cost coefficients. Expected keys:
          - 'wait_penalty'  (float): cost per car per minute waiting
          - 'bay_cost'      (float): cost per open bay per period

    Returns
    -------
    float
        Cost incurred this period.
    """
    wait_penalty = params.get("wait_penalty", 1.0)
    bay_cost = params.get("bay_cost", 0.5)

    waiting_cost = state.queue_length * wait_penalty
    operating_cost = decision.get("bays_open", state.bays_open) * bay_cost

    return waiting_cost + operating_cost
