"""
Main script for running Maximum Entropy Model.


Katerina Capouskova 2019, kcapouskova@hotmail.com
"""
import argparse

from data_preprocess import mean_over_areas, binrize, number_of_patterns, \
    average_activity, pairwise_covar, empirical_pairwise_activation, \
    entropy_of_distribution_data, entropy_model
from max_ent import calculate_prob_max_ent
from utilities import create_dir
from visualizations import sequence_count_plot, sequence_count_plot_model


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run Max Ent pipeline.')

    parser.add_argument('--input', type=str, help='Path to the input directory',
                        required=True)
    parser.add_argument('--threshold', type=float, help='Threshold for data to be '
                                                      'active and inactive.',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--areas', type=list,
                        help='List of list of indicating higer scale areas.',
                        required=False)
    return parser.parse_args()


def main():
    """
    Max Ent Model on fMRI data
    """
    args = parse_args()
    input_path = args.input
    output_path = args.output
    brain_areas = args.areas
    threshold = args.threshold

    brain_areas = [[21, 3, 19, 24, 16, 20, 18, 17, 62, 40, 45, 46, 41, 49, 44, 48, 47, 25], [36, 50, 15, 29],
               [51, 58, 34, 60, 56, 55, 14, 7, 10, 9], [2, 12, 8, 5, 13, 11, 4, 63, 53, 54, 57, 52, 61],
               [37, 39, 59, 38, 6, 26, 28, 27], [30, 35, 23, 22, 32, 31, 0, 1, 64, 42, 43, 33, 65]]

    lambdas_dict = {'lambda1': -4.37389, 'lambda2': -0.0181369, 'lambda3':
        0.0434916, 'lambda4': 0.0138453, 'lambda5': -0.182505, 'lambda6': 0.0141578,
        'lambda7': 0.0448107, 'lambda8': -0.143712, 'lambda9': -0.0357837,
        'lambda10': -0.0992271, 'lambda11': -0.0599827, 'lambda12': -0.194269,
        'lambda13': -0.0807454, 'lambda14': -0.0203112, 'lambda15': 0.205356,
        'lambda16': -0.0932409,'lambda17': -0.286996, 'lambda18': -0.0666362,
        'lambda19': 0.047215, 'lambda20': -0.0219382, 'lambda21': 0.0555467,
        'lambda22': -0.417749}

    create_dir(output_path)

    array = mean_over_areas(input_path, brain_areas, output_path)
    binarized = binrize(array, threshold)
    unique, count, patterns, count_norm = number_of_patterns(binarized)
    entropy_of_distribution_data(count_norm, output_path)
    sequence_count_plot(patterns, count_norm, output_path)
    areas_average = average_activity(binarized, output_path)
    empirical_pairwise_activation(binarized, output_path)
    pairwise_covar(binarized, areas_average, output_path)
    probabilities, probas_list = calculate_prob_max_ent(lambdas_dict, output_path)
    sequence_count_plot_model(patterns, count_norm, probabilities, output_path)
    entropy_model(probas_list, output_path)


if __name__ == '__main__':
    main()