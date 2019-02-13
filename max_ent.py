"""
Script for calculating probabilities as modeled by Max Ent model.
Lambda are provided from a script written in Mathematica.


Katerina Capouskova 2019, kcapouskova@hotmail.com
"""
import itertools
import json
import os

import numpy as np


def calculate_prob_max_ent(lambdas_dict, output_path):
    """
    Calculates the probability of a pattern as given my Max Ent model.

    :param lambdas_dict: dictionary of lambdas
    :type lambdas_dict: dict
    :param output_path: path to output directory
    :type output_path: str
    :return probabilities, probas_list: probabilites of pattern, pobas list
    :rtype probabilities, probas_list: dict, []
    """
    patterns = list(itertools.product([0, 1], repeat=6))
    probabilities = {}
    probas_list = []
    for pattern in patterns:
        p = np.exp(lambdas_dict['lambda1'] + lambdas_dict['lambda2'] * pattern[0]
                   + lambdas_dict['lambda3'] * pattern[1] - lambdas_dict['lambda8']
                   * pattern[0] * pattern[1] + lambdas_dict['lambda4'] * pattern[2] -
                   lambdas_dict['lambda9'] * pattern[0] * pattern[2] -
                   lambdas_dict['lambda13'] * pattern[1] * pattern[2] +
                   lambdas_dict['lambda5'] * pattern[3] - lambdas_dict['lambda10']
                   * pattern[0] * pattern[3] - lambdas_dict['lambda14'] *
                   pattern[1] * pattern[3] - lambdas_dict['lambda17'] *
                   pattern[2] * pattern[3] + lambdas_dict['lambda6'] * pattern[4]
                   + lambdas_dict['lambda11'] * pattern[0] * pattern[4] +
                   lambdas_dict['lambda15'] * pattern[1] * pattern[4] -
                   lambdas_dict['lambda18'] * pattern[2] * pattern[4] +
                   lambdas_dict['lambda20'] * pattern[3] * pattern[4] +
                   lambdas_dict['lambda7'] * pattern[5] - lambdas_dict['lambda12']
                   * pattern[0] * pattern[5] - lambdas_dict['lambda16'] *
                   pattern[1] * pattern[5] + lambdas_dict['lambda19'] * pattern[2]
                   * pattern[5] - lambdas_dict['lambda21'] * pattern[3] * pattern[5] -
                   lambdas_dict['lambda22'] * pattern[4] * pattern[5])

        u = [int(i) for i in pattern]
        pattern = ''.join([str(i) for i in u])
        pattern = 's{}'.format(pattern)
        probabilities.update({str(pattern): p})
        probas_list.append(p)
    with open(os.path.join(output_path, 'model_probabilities.json'), 'w') as fp:
        json.dump(probabilities, fp)
    return probabilities, probas_list
