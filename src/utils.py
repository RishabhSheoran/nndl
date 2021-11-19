import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def get_error(outputs, labels, batch_size):
    lab = torch.where(outputs >= 0.5, 1, 0)
    indicator = torch.where(lab != labels, 1, 0)
    non_matches = torch.sum(indicator, axis=0)
    error = non_matches.float()/batch_size
    error = error.sum() / 6
    return error.item() * 100


def get_accuracy(outputs, labels, batch_size):
    pred = torch.where(outputs >= 0.5, 1, 0)
    indicator = torch.where(pred == labels, 1, 0)
    non_matches = torch.sum(indicator, axis=0)
    acc = non_matches.float()/batch_size
    acc = acc.sum() / 6
    return acc.item() * 100


# outputs_np_arr and labels_np_arr are numpy arrays
def get_f1_score(labels_np_arr, outputs_np_arr):
    f_score = f1_score(labels_np_arr, outputs_np_arr >= 0.5, average="samples")
    return f_score

# outputs_np_arr and labels_np_arr are numpy arrays


def get_precision_score(labels_np_arr, outputs_np_arr):
    precision = precision_score(
        labels_np_arr, outputs_np_arr >= 0.5, average="samples")
    return precision

# outputs_np_arr and labels_np_arr are numpy arrays


def get_recall_score(labels_np_arr, outputs_np_arr):
    recall = recall_score(labels_np_arr, outputs_np_arr >=
                          0.5, average="samples")
    return recall


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param/1e6)
    )
