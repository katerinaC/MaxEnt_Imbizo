"""
Utilities file for different operations to be used in the processing scripts.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import os

import numpy as np


def return_paths_list(input_path, pattern):
    """
    Loads the .mat file and saves it as a .csv file or files. Returns all
    paths in a directory with specific format as a list.

    :param input_path: path to the files directory
    :type input_path: str
    :param pattern: the pattern of files to include
    :type pattern: str
    :return: list of all paths in a directory
    :rtype: []
    """
    # list of all .csv files in the directory
    paths_list = []
    if pattern == '.csv' or pattern == '.npz':
        for directory, _, files in os.walk(input_path):
            paths_list += [os.path.join(directory, file) for file in files
                           if file.endswith(pattern)]
    return paths_list


def trasform_data(input_path, output_path, n_subjects, n_tasks):
    """
    Loads data in npy format and outputs them in a desired format.
    For each subject get Time X Brain Areas matrix as a .csv file

    :param input_path: path to the file
    :type input_path: str
    :param output_path: path where to save the .csv file/s
    :type output_path: str
    :param n_subjects: number of subjects
    :type n_subjects: int
    :param n_tasks: number of tasks
    :type n_tasks: int
    """
    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)
    data = np.load(input_path)
    data = np.swapaxes(data, 2, 3)
    print data.shape
    for subject in range(n_subjects):
        for task in range(n_tasks):
            np.savetxt(os.path.join(output_path, 'subject{}_task{}.csv'.format
            (subject, task)), data[subject, task, :, :], delimiter=',')


def create_new_output_path(input_path, output_path):
    """
    Returns a new output_path from input_path

    :param input_path: path to the files dir
    :type input_path: str
    :param output_path: path where to save the file
    :type output_path: str
    :return: new output path
    :rtype: str
    """
    base_name = os.path.basename(input_path)
    return os.path.join(output_path, base_name)


def create_dir(output_path):
    """
    Creates a new directory for output path

    :param output_path: path to output dir
    :type output_path: str
    """
    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)
