"""
Visualization tools for Max Ent Model.

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid", context="paper")


def sequence_count_plot(unique, count, output_path):
    """
    Plots the unique patterns occurences.

    :param unique: unique patterns
    :type unique: np.ndarray
    :param: unique count, count of the unique patterns
    :type: unique count: np.ndarray
    :param: output_path: path to the output folder
    :type: output_path: str
    """
    counts, patterns = (list(t) for t in zip(*sorted(zip(count.tolist(), unique))))
    sns.barplot(x=patterns, y=counts, palette='plasma')
    plt.axhline(0, color='k', clip_on=False)
    plt.xticks(rotation=90)
    plt.ylabel('Count of unique patterns')
    plt.xlabel('Patterns')
    sns.despine(left=True)
    plt.savefig(os.path.join(output_path, 'Unique_patterns_counts.png'))


def sequence_count_plot_model(unique, count, probabilities, output_path):
    """
    Plots the unique patterns occurences and model prob values

    :param unique: unique patterns
    :type unique: np.ndarray
    :param: unique count, count of the unique patterns
    :type: unique count: np.ndarray
    :param: output_path: path to the output folder
    :type: output_path: str
    """
    df = pd.DataFrame([count], columns=unique, index=['data'])
    prob = pd.DataFrame(probabilities, index=['prob'])
    df_merge = pd.concat([df, prob], axis=0, join='outer')
    df_merge.dropna(inplace=True)
    #df_norm = StandardScaler().fit_transform(df_merge)
    #df_norm = pd.DataFrame([df_norm], columns=unique, index=['data', 'prob'])
    df_merge_T = df_merge.T
    df_sorted_T = df_merge_T.sort_values(by=['data'])
    df_sorted = df_sorted_T.T
    fig, ax1 = plt.subplots(figsize=(15, 10))
    df_sorted.loc['data', :].plot(kind='bar', color='y')
    df_sorted.loc['prob', :].plot(kind='line', marker='d', secondary_y=False)
    ax1.tick_params(axis='x', labelrotation=90)
    plt.savefig(os.path.join(output_path, 'Real_and_model_plot.png'))

