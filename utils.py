import ast
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_vec(item, target_length=20):
    one_array = np.array(ast.literal_eval(item))

    if len(one_array) < target_length:
        padding_length = target_length - len(one_array)
        one_array = np.pad(one_array, (0, padding_length), 'constant', constant_values=0)

    elif len(one_array) > target_length:
        one_array = one_array[:target_length]

    return one_array


def get_len(item):
    sen_len = len(item.split(' '))
    return sen_len


def format_item(csv_file):
    pd_all = pd.read_csv(csv_file).sample(frac=1).reset_index(drop=True)
    pd_all.columns = ['file_path', 'description', 'category', 'score', 'distribution', 'ground_truth']
    label_train_list = LabelEncoder().fit_transform(pd_all['ground_truth'])
    pd_all['label'] = label_train_list
    pd_all['len'] = pd_all['description'].map(get_len)

    return pd_all


def generate_item(csv_file):
    return (len(csv_file), list(csv_file['description']), list(csv_file['label']), list(csv_file['file_path']),
            list(csv_file['distribution'].map(get_vec)), list(csv_file['score']),)


def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)
