import numpy as np

def filter_matrix(matrix_to_filter, base_rates, n_highest, low_act_frequency, magnitude_factor):
    # single dimension based filters
    features_to_exclude = np.array([], dtype=int)
    if n_highest is not None:
        high_base_rate_indices = high_base_rate_filter(base_rates, n_highest=n_highest)
        features_to_exclude = np.union1d(features_to_exclude, high_base_rate_indices)

    if low_act_frequency is not None:
        low_base_rate_indices = low_base_rate_filter(base_rates, threshold=low_act_frequency)
        features_to_exclude = np.union1d(features_to_exclude, low_base_rate_indices)

    if len(features_to_exclude) > 0:
        matrix_to_filter[features_to_exclude, :] = 0
        matrix_to_filter[:, features_to_exclude] = 0

    # two dimension based filters
    if magnitude_factor is not None:
        magnitude_filter_indices = magnitude_filter(base_rates, magnitude_factor=magnitude_factor)
        matrix_to_filter[magnitude_filter_indices] = 0

    return matrix_to_filter
        

def filter_original_sae_ids(base_rates, n_highest, low_act_frequency, magnitude_factor):
    original_sae_ids_to_exclude = set()
    if n_highest is not None:
        high_base_rate_indices = high_base_rate_filter(base_rates, n_highest=n_highest)
        original_sae_ids_to_exclude.update(high_base_rate_indices)

    if low_act_frequency is not None:
        low_base_rate_indices = low_base_rate_filter(base_rates, threshold=low_act_frequency)
        original_sae_ids_to_exclude.update(low_base_rate_indices)

    if magnitude_factor is not None:
        magnitude_filter_indices = magnitude_filter(base_rates, magnitude_factor=magnitude_factor)
        original_sae_ids_to_exclude.update(np.where(magnitude_filter_indices)[0])

    return original_sae_ids_to_exclude

def low_base_rate_filter(base_rates, threshold):
    low_base_rate_mask = base_rates < threshold
    low_base_rate_indices = np.where(low_base_rate_mask)[0]
    return low_base_rate_indices

def high_base_rate_filter(base_rates, n_highest):
    highest_base_rate_indices = np.argsort(base_rates)[-n_highest:][::-1]
    return highest_base_rate_indices

def magnitude_filter(base_rates, magnitude_factor):
    # create a matrix that divedes each entry pairwise
    rate_matrix = base_rates[:, np.newaxis] / base_rates[np.newaxis, :]
    # replace np.nan values with 1
    rate_matrix = np.nan_to_num(rate_matrix, nan=0.0)
    # create filter where values are either > MAGNITUDE_FACTOR or < (1 / MAGNITUDE_FACTOR)
    magnitude_filter_indices = (rate_matrix > magnitude_factor) | (rate_matrix < (1.0 / magnitude_factor))
    return magnitude_filter_indices



