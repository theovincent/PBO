import numpy as np
from scipy.stats import t as student_variable


def confidence_interval(mean, std, n_samples, confidence_level=0.95):
    t_crit = np.abs(student_variable.ppf((1 - confidence_level) / 2, n_samples - 1))
    lower_bound = mean - t_crit * (std / np.sqrt(n_samples))
    upper_bound = mean + t_crit * (std / np.sqrt(n_samples))

    return lower_bound, upper_bound


def means_and_confidence_interval(metrics):
    means = np.mean(metrics, axis=0)
    stds = np.std(metrics, axis=0)
    n_seeds = metrics.shape[0]
    confidence_intervals = np.zeros((2, metrics.shape[1]))

    for idx_element in range(metrics.shape[1]):
        confidence_intervals[0, idx_element], confidence_intervals[1, idx_element] = confidence_interval(
            means[idx_element], stds[idx_element], n_seeds
        )

    return means, confidence_intervals
