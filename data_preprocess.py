"""
Data preprocessing script.


Katerina Capouskova 2019, kcapouskova@hotmail.com
"""
import itertools
import json
import os

import numpy as np

from utilities import return_paths_list


def mean_over_areas(input_path, brain_areas, output_path):
    """
    Takes BOLD raw data a means over selected higher level areas

    :param input_path: path to input files
    :type input_path: str
    :param output_path: path to output directory
    :type output_path: str
    :param brain_areas: specified brain areas
    :type brain_areas: [[]]
    :return: array with all subjects and time points in one task
    :rtype: np.ndarray
    """
    paths = return_paths_list(input_path, '.csv')
    arr = np.loadtxt(paths[0], delimiter=',')
    subjects = len(paths)
    time_points = arr.shape[0]
    all_time_p = time_points*subjects
    concat = np.zeros((all_time_p, 6))
    start = 0
    for path in paths:
        array = np.loadtxt(path, delimiter=',')
        array_selected = np.zeros((time_points, 6))
        for i in range(6):
            select = np.squeeze(array[:, [brain_areas[i]]])
            array_selected[:, i] = np.mean(select, axis=1)
        concat[start: (start+time_points), :] = array_selected
        start += time_points
    np.save(os.path.join(output_path, 'lobes_concat.npy'), concat)
    return concat


def binrize(array, threshold):
    """
    Binarizes raw fMRI data according a given threshold.

    :param array: input array to be binarized
    :type array: np.ndarray
    :param threshold: threshold for binarization
    :type threshold: float
    :return: binarized array with all subjects and time points in one task
    :rtype: np.ndarray
    """
    array[array > threshold] = 1
    array[array <= threshold] = -1
    return array


def number_of_patterns(bin_array):
    """
    Calculates the unique patterns occurences

    :param bin_array: binarize input array
    :type bin_array: np.ndarray
    :return: unique patterns, count, unique patterns for plot, norm counts
    :rtype: np.ndarray, np.ndarray, [], np.ndarray
    """
    patterns = []
    unique, count = np.unique(bin_array, axis=0, return_counts=True)
    for u in unique:
        u = u.tolist()
        u = [int(i) for i in u]
        pattern = ''.join([str(i) if i==1 else str(0) for i in u])
        pattern = 's{}'.format(pattern)
        patterns.append(pattern)
    count_norm = count / float(np.sum(count))
    return unique, count, patterns, count_norm


def entropy_of_distribution_data(count_norm, output_path):
    """
    Entropy of data distribution

    :param count_norm: proabilities of states
    :type count_norm: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :return: entropy
    :rtype: float
    """
    count_norm = np.expand_dims(count_norm, axis=1)
    entropy = -np.sum(count_norm * np.log2(count_norm))
    ent = {'entropy': entropy}
    with open(os.path.join(output_path, 'entropy.json'), 'w') as fp:
        json.dump(ent, fp)
    return entropy


def entropy_model(model_probabilities, output_path):
    """
     Entropy of model distribution

     :param model_probabilities: proabilities of states
     :type model_probabilities: np.ndarray
     :param output_path: path to output directory
     :type output_path: str
     :return: entropy
     :rtype: float
     """
    entropy = -np.sum(model_probabilities * np.log2(model_probabilities))
    ent = {'model_entropy': entropy}
    with open(os.path.join(output_path, 'model_entropy.json'), 'w') as fp:
        json.dump(ent, fp)
    return entropy


def average_activity(bin_array, output_path):
    """
    Average activity of each brain area

    :param bin_array: binarize input array
    :type bin_array: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :return: average activity of each area
    :rtype: dict
    """
    time_points, brain_areas = bin_array.shape
    areas_average = {}
    areas_names = ['prefrontal', 'motor', 'parietal', 'temporal', 'occipital',
                   'limbic']
    for area in range(brain_areas):
        areas_average.update({areas_names[area]: np.mean(bin_array[:, area])})
    with open(os.path.join(output_path, 'average_activities.json'), 'w') as fp:
        json.dump(areas_average, fp)
    return areas_average


def empirical_pairwise_activation(bin_array, output_path):
    """
    Empirical pairwise joint activation rate of two regions

    :param bin_array: binarize input array
    :type bin_array: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :return: empirical pairwise interaction
    :rtype: []
    """
    pairwise_activations = {}
    areas_names = ['prefrontal', 'motor', 'parietal', 'temporal', 'occipital',
                   'limbic']
    time_points, brain_areas = bin_array.shape
    for i, j in itertools.combinations(areas_names, 2):
        pair_ij = []
        for t in range(time_points):
            pairwise_act = bin_array[t, areas_names.index(i)] * bin_array[
                t, areas_names.index(j)]
            pair_ij.append(pairwise_act)
        final_ij = sum(pair_ij)/time_points
        pairwise_activations.update({'{}_{}'.format(i, j): final_ij})
    with open(os.path.join(output_path, 'pairwise_activations.json'), 'w') as fp:
        json.dump(pairwise_activations, fp)
    return pairwise_activations


def pairwise_covar(bin_array, areas_average, output_path):
    """
    Empirical pairwise covariance

    :param bin_array: binarize input array
    :type bin_array: np.ndarray
    :param areas_average: average activity of each area
    :type areas_average: dict
    :param output_path: path to output directory
    :type output_path: str
    :return: empirical pairwise covariance
    :rtype: {}
    """
    pairwise_covars = {}
    areas_names = ['prefrontal', 'motor', 'parietal', 'temporal', 'occipital',
                   'limbic']
    time_points, brain_areas = bin_array.shape
    for i, j in itertools.combinations(areas_names, 2):
        covs = []
        for t in range(time_points):
            cov = (bin_array[t, areas_names.index(i)] - areas_average[i]) * (
                    bin_array[t, areas_names.index(j)] - areas_average[j])
            covs.append(cov)
        final_cov = sum(covs) / time_points
        pairwise_covars.update({'{}_{}'.format(i, j): final_cov})
    with open(os.path.join(output_path, 'pairwise_covars.json'), 'w') as fp:
        json.dump(pairwise_covars, fp)
    return pairwise_covars
