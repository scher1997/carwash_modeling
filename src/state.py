from dataclasses import dataclass, field
from typing import List


@dataclass
class State:
    """
    S_t: the state variable at time t.

    Holds everything known at time t that is relevant to future decisions.
    Add or remove fields as the model grows — keep this the single source
    of truth for system state.
    """
    time: float = 0.0           # simulation clock (minutes)
    queue_length: int = 0       # cars waiting
    bays_open: int = 1          # number of active service bays
    bays_busy: int = 0          # bays currently serving a car
    cars_served: int = 0        # cumulative completions
    total_wait: float = 0.0     # cumulative wait time across all cars

    @property
    def bays_available(self) -> int:
        return self.bays_open - self.bays_busy

    @property
    def avg_wait(self) -> float:
        if self.cars_served == 0:
            return 0.0
        return self.total_wait / self.cars_served
