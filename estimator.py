from functools import total_ordering

import numpy as np


def lerp(a, b, t):
    return a + (b - a) * t


@total_ordering
class Abs_Coef_Estimator_at_freq:
    def __init__(
        self,
        frequency: float,
        lower_bound: float,
        upper_bound: float,
    ):
        self.frequency = frequency
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def unif(self):
        return np.random.uniform(self.lower_bound, self.upper_bound)

    def __eq__(self, other):
        return self.frequency == other.frequency

    def __lt__(self, other):
        return self.frequency < other.frequency


class Abs_Coef_Estimator:
    def __init__(self, freq_dep_estimators: list[Abs_Coef_Estimator_at_freq]):
        self.freq_dep_estimators = sorted(freq_dep_estimators)
        self.min_frequency: float = self.freq_dep_estimators[0].frequency
        self.max_frequency: float = self.freq_dep_estimators[-1].frequency

    def estimate_abs_coef(self, frequency):
        if frequency < self.min_frequency:
            return self.freq_dep_estimators[0].unif()
        if frequency > self.max_frequency:
            return self.freq_dep_estimators[-1].unif()

        for i in range(len(self.freq_dep_estimators)):
            if frequency > self.freq_dep_estimators[i + 1].frequency:
                continue
            bottom_freq = self.freq_dep_estimators[i].frequency
            top_freq = self.freq_dep_estimators[i + 1].frequency

            t = (frequency - bottom_freq) / (top_freq - bottom_freq)

            lerped_upper = lerp(
                self.freq_dep_estimators[i].upper_bound,
                self.freq_dep_estimators[i + 1].upper_bound,
                t,
            )
            lerped_lower = lerp(
                self.freq_dep_estimators[i].lower_bound,
                self.freq_dep_estimators[i + 1].lower_bound,
                t,
            )
            return np.random.uniform(low=lerped_lower, high=lerped_upper)


class Ceiling_Estimator(Abs_Coef_Estimator):
    def __init__(self):
        freq_dep_estimators = [
            Abs_Coef_Estimator_at_freq(
                frequency=125, lower_bound=0.01, upper_bound=0.7
            ),
            Abs_Coef_Estimator_at_freq(frequency=250, lower_bound=0.15, upper_bound=1),
            Abs_Coef_Estimator_at_freq(frequency=500, lower_bound=0.4, upper_bound=1),
            Abs_Coef_Estimator_at_freq(frequency=2000, lower_bound=0.4, upper_bound=1),
            Abs_Coef_Estimator_at_freq(frequency=4000, lower_bound=0.3, upper_bound=1),
        ]
        super().__init__(freq_dep_estimators=freq_dep_estimators)


class Floor_Estimator(Abs_Coef_Estimator):
    def __init__(self):
        freq_dep_estimators = [
            Abs_Coef_Estimator_at_freq(
                frequency=125, lower_bound=0.01, upper_bound=0.2
            ),
            Abs_Coef_Estimator_at_freq(
                frequency=250, lower_bound=0.01, upper_bound=0.3
            ),
            Abs_Coef_Estimator_at_freq(
                frequency=500, lower_bound=0.05, upper_bound=0.5
            ),
            Abs_Coef_Estimator_at_freq(
                frequency=1000, lower_bound=0.15, upper_bound=0.6
            ),
            Abs_Coef_Estimator_at_freq(
                frequency=2000, lower_bound=0.25, upper_bound=0.75
            ),
            Abs_Coef_Estimator_at_freq(
                frequency=4000, lower_bound=0.3, upper_bound=0.8
            ),
        ]
        super().__init__(freq_dep_estimators=freq_dep_estimators)


class Wall_Estimator(Abs_Coef_Estimator):
    def __init__(self):
        freq_dep_estimators = [
            Abs_Coef_Estimator_at_freq(
                frequency=125, lower_bound=0.01, upper_bound=0.5
            ),
            Abs_Coef_Estimator_at_freq(
                frequency=250, lower_bound=0.01, upper_bound=0.5
            ),
            Abs_Coef_Estimator_at_freq(
                frequency=500, lower_bound=0.01, upper_bound=0.3
            ),
            Abs_Coef_Estimator_at_freq(
                frequency=1000, lower_bound=0.01, upper_bound=0.12
            ),
        ]
        super().__init__(freq_dep_estimators=freq_dep_estimators)
