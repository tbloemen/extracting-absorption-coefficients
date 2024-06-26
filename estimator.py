"""
Houses all the files needed for the estimator of absorption profiles.
"""

from functools import total_ordering

import numpy as np
from scipy.stats import truncnorm


def log_lerp(x0: float, f0: float, x1: float, f1: float, x: float) -> float:
    """
    Interpolates the value between x0 and x1 according to the logarithmic scale.
    :param x0: the known x0 value where x0 <= x.
    :param f0: the associated f(x0).
    :param x1: the known x1 value where x <= x1.
    :param f1: the associated f(x1).
    :param x: the value where we want to estimate f(x) from.
    :return: The interpolated f(x).
    """
    return (f1 - f0) * np.log(x / x0) / np.log(x1 / x0) + f0


@total_ordering
class AbsCoefEstimatorAtFreq:
    """
    Helper class for AbsCoefEstimator. It contains the mean, standard deviation, upper and lower bound of a dataset
    at a certain frequency.
    """

    def __init__(
        self,
        frequency: float,
        mean: float,
        stdev: float,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        self.frequency = frequency
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.mean = mean
        self.stdev = stdev

    def __eq__(self, other) -> bool:
        return self.frequency == other.frequency

    def __lt__(self, other) -> bool:
        return self.frequency < other.frequency

    def trunc_norm(self) -> float:
        """
        Samples a random nummer according to the truncated normal distribution.
        :return: A float according to the truncated normal distribution.
        """
        a, b = (self.lower_bound - self.mean) / self.stdev, (
            self.upper_bound - self.mean
        ) / self.stdev
        rv = truncnorm(a, b, loc=self.mean, scale=self.stdev)
        return rv.rvs()

    def unif(self) -> float:
        """
        Samples a random nummer according to the uniform distribution.
        :return: A float according to the uniform distribution.
        """
        return np.random.uniform(self.lower_bound, self.upper_bound)


def lerp_estimator(
    a: AbsCoefEstimatorAtFreq, b: AbsCoefEstimatorAtFreq, t: float
) -> AbsCoefEstimatorAtFreq:
    """
    Returns a new AbsCoefEstimator at frequency t.
    :param a: an AbsCoefEstimatorAtFreq where the frequency <= t.
    :param b: an AbsCoefEstimatorAtFreq where the frequency >= t.
    :param t: the frequency at which to estimate the AbsCoefEstimatorAtFreq.
    :return: an AbsCoefEstimatorAtFreq which is logarithmically interpolated between a and b.
    """
    lower_bound, upper_bound = None, None

    if a.lower_bound is not None:
        lower_bound = log_lerp(
            x0=a.frequency, x1=b.frequency, f0=a.lower_bound, f1=b.lower_bound, x=t
        )
    if a.upper_bound is not None:
        upper_bound = log_lerp(
            x0=a.frequency, x1=b.frequency, f0=a.upper_bound, f1=b.upper_bound, x=t
        )
    mean = log_lerp(x0=a.frequency, x1=b.frequency, f0=a.mean, f1=b.mean, x=t)
    stdev = log_lerp(x0=a.frequency, x1=b.frequency, f0=a.stdev, f1=b.stdev, x=t)
    return AbsCoefEstimatorAtFreq(
        frequency=t,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        mean=mean,
        stdev=stdev,
    )


class AbsCoefEstimator:
    """
    The main class to estimate the absorption coefficient at frequencies other than the center frequency.
    """

    def __init__(self, freq_dep_estimators: list[AbsCoefEstimatorAtFreq]):
        self.freq_dep_estimators = sorted(freq_dep_estimators)
        self.min_frequency: float = self.freq_dep_estimators[0].frequency
        self.max_frequency: float = self.freq_dep_estimators[-1].frequency

    def estimate_abs_coef(self, frequency):
        """
        Returns the absorption coefficient sampled by the truncated norm at the given frequency.
        :param frequency: The frequency at which the absorption coefficient is needed.
        :return: The sampled absorption coefficient.
        """
        return self.get_estimator_at_freq(frequency=frequency).trunc_norm()

    def get_estimator_at_freq(self, frequency) -> AbsCoefEstimatorAtFreq:
        """
        Returns an estimator at the given frequency.
        :param frequency: the frequency.
        :return: an AbsCoefEstimatorAtFreq instance.
        """
        if frequency < self.min_frequency:
            return AbsCoefEstimatorAtFreq(
                frequency=frequency,
                lower_bound=self.freq_dep_estimators[0].lower_bound,
                upper_bound=self.freq_dep_estimators[0].upper_bound,
                mean=self.freq_dep_estimators[0].mean,
                stdev=self.freq_dep_estimators[0].stdev,
            )
        if frequency > self.max_frequency:
            return AbsCoefEstimatorAtFreq(
                frequency=frequency,
                lower_bound=self.freq_dep_estimators[-1].lower_bound,
                upper_bound=self.freq_dep_estimators[-1].upper_bound,
                mean=self.freq_dep_estimators[-1].mean,
                stdev=self.freq_dep_estimators[-1].stdev,
            )

        for i in range(len(self.freq_dep_estimators)):
            if frequency > self.freq_dep_estimators[i + 1].frequency:
                continue

            return lerp_estimator(
                self.freq_dep_estimators[i], self.freq_dep_estimators[i + 1], frequency
            )


class ReflectiveEstimator(AbsCoefEstimator):
    def __init__(self):
        freq_dep_estimators = [
            AbsCoefEstimatorAtFreq(
                frequency=125,
                lower_bound=0.01,
                upper_bound=0.35,
                mean=0.121129032,
                stdev=0.106941149,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=250,
                lower_bound=0.01,
                upper_bound=0.20,
                mean=0.083750000,
                stdev=0.063491231,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=500,
                lower_bound=0.01,
                upper_bound=0.2,
                mean=0.065555556,
                stdev=0.042112667,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=1000,
                lower_bound=0.01,
                upper_bound=0.10,
                mean=0.056071429,
                stdev=0.026811027,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=2000,
                lower_bound=0.01,
                upper_bound=0.15,
                mean=0.055396825,
                stdev=0.030151520,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=4000,
                lower_bound=0.01,
                upper_bound=0.20,
                mean=0.057096774,
                stdev=0.033406934,
            ),
        ]
        super().__init__(freq_dep_estimators)


class CeilingEstimator(AbsCoefEstimator):
    def __init__(self):
        freq_dep_estimators = [
            AbsCoefEstimatorAtFreq(
                frequency=125,
                lower_bound=0.02,
                upper_bound=0.65,
                mean=0.237631579,
                stdev=0.143998782,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=250,
                lower_bound=0.05,
                upper_bound=1,
                mean=0.506515152,
                stdev=0.262012874,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=500,
                lower_bound=0.14,
                upper_bound=1,
                mean=0.676578947,
                stdev=0.273079000,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=1000,
                lower_bound=0.05,
                upper_bound=1,
                mean=0.724393939,
                stdev=0.271310942,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=2000,
                lower_bound=0.05,
                upper_bound=1,
                mean=0.705526316,
                stdev=0.286567710,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=4000,
                lower_bound=0.05,
                upper_bound=1,
                mean=0.704400000,
                stdev=0.278112557,
            ),
        ]
        super().__init__(freq_dep_estimators=freq_dep_estimators)


class FloorEstimator(AbsCoefEstimator):
    def __init__(self):
        freq_dep_estimators = [
            AbsCoefEstimatorAtFreq(
                frequency=125,
                lower_bound=0.01,
                upper_bound=0.65,
                mean=0.226382979,
                stdev=0.153047670,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=250,
                lower_bound=0.02,
                upper_bound=1,
                mean=0.466444444,
                stdev=0.308825896,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=500,
                lower_bound=0.05,
                upper_bound=1,
                mean=0.673541667,
                stdev=0.305401651,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=1000,
                lower_bound=0.15,
                upper_bound=1,
                mean=0.716444444,
                stdev=0.282482144,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=2000,
                lower_bound=0.10,
                upper_bound=1,
                mean=0.758750000,
                stdev=0.258791175,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=4000,
                lower_bound=0.15,
                upper_bound=1,
                mean=0.784893617,
                stdev=0.225194335,
            ),
        ]
        super().__init__(freq_dep_estimators=freq_dep_estimators)


class WallEstimator(AbsCoefEstimator):
    def __init__(self):
        freq_dep_estimators = [
            AbsCoefEstimatorAtFreq(
                frequency=125,
                lower_bound=0.02,
                upper_bound=0.65,
                mean=0.212564103,
                stdev=0.142431901,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=250,
                lower_bound=0.03,
                upper_bound=1,
                mean=0.425000000,
                stdev=0.266805893,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=500,
                lower_bound=0.1,
                upper_bound=1,
                mean=0.580256410,
                stdev=0.300518921,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=1000,
                lower_bound=0.05,
                upper_bound=1,
                mean=0.632121212,
                stdev=0.306937805,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=2000,
                lower_bound=0.05,
                upper_bound=1,
                mean=0.637435897,
                stdev=0.313991628,
            ),
            AbsCoefEstimatorAtFreq(
                frequency=4000,
                lower_bound=0.05,
                upper_bound=1,
                mean=0.641282051,
                stdev=0.309936562,
            ),
        ]
        super().__init__(freq_dep_estimators=freq_dep_estimators)
