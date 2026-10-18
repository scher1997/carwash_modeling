from .state import State

# Placeholder economics ($) — calibrate to the real site before trusting results.
DEFAULT_COST_PARAMS = {
    "price_per_wash": 12.0,  # revenue per completed wash
    "cost_per_wash": 3.0,    # variable cost per wash (water, chemicals, power)
    "bay_cost": 0.5,         # cost per staffed bay per minute ($30/hr labour + energy)
    "wait_penalty": 0.2,     # goodwill cost per car per minute in queue
}


def contribution(state: State, decision: dict, params: dict, exogenous: dict = None) -> dict:
    """
    C(S_t, x_t, W_{t+1}): economics of one time step.

    Revenue and variable cost accrue when a wash completes. A bay costs money
    while it is staffed: that is every open bay, plus any closing bay still
    finishing its last car.

    Parameters
    ----------
    state : State
        Current state S_t (before transition).
    decision : dict
        Decision x_t.
    params : dict
        Overrides for DEFAULT_COST_PARAMS.
    exogenous : dict, optional
        Realized W_{t+1}; uses 'completions'.

    Returns
    -------
    dict
        revenue, wash_cost, bay_cost, wait_cost, profit, and cost (= -profit,
        the quantity policies and tuning minimize).
    """
    p = {**DEFAULT_COST_PARAMS, **(params or {})}
    completions = (exogenous or {}).get("completions", 0)
    staffed_bays = max(decision.get("bays_open", state.bays_open), state.bays_busy)

    revenue = completions * p["price_per_wash"]
    wash_cost = completions * p["cost_per_wash"]
    bay_cost = staffed_bays * p["bay_cost"]
    wait_cost = state.queue_length * p["wait_penalty"]
    profit = revenue - wash_cost - bay_cost - wait_cost

    return {
        "revenue": revenue,
        "wash_cost": wash_cost,
        "bay_cost": bay_cost,
        "wait_cost": wait_cost,
        "profit": profit,
        "cost": -profit,
    }


def cost(state: State, decision: dict, params: dict, exogenous: dict = None) -> float:
    """Net cost of one step (= -profit). Lower is better."""
    return contribution(state, decision, params, exogenous)["cost"]
