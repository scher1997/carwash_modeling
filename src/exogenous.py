import numpy as np


class ExogenousInfo:
    """
    W_{t+1}: exogenous information that arrives after a decision is made.

    Encapsulates all randomness in the model. Swap out distributions here
    when calibrating to real data — nothing else needs to change.
    """

    def __init__(self, arrival_rate: float = 10.0, service_rate: float = 8.0, seed: int = None):
        """
        Parameters
        ----------
        arrival_rate : float
            Mean cars per hour.
        service_rate : float
            Mean cars served per hour per bay.
        seed : int, optional
            RNG seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.arrival_rate = arrival_rate    # cars/hour
        self.service_rate = service_rate    # cars/hour/bay

    def next_interarrival(self) -> float:
        """Minutes until the next car arrives (exponential / Poisson process)."""
        return self.rng.exponential(60.0 / self.arrival_rate)

    def next_service_time(self) -> float:
        """Minutes to wash one car in one bay."""
        return self.rng.exponential(60.0 / self.service_rate)
