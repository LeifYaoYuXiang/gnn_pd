import torch
from scipy import sparse
import numpy as np
import dgl
import os
from dgl.data.utils import save_graphs
from sklearn.preprocessing import MinMaxScaler
import scipy.io as scio

chosen_file_index = [1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 30, 31,
                     32, 33, 34, 35, 36, 37, 39, 40, 41, 42,43, 44, 45, 46,
                     47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62]


def load_mat_filepath_from_dir(dir_path, chosen_file_index):
    mat_filepath_list = []
    mat_name_list = []
    filename_list = os.listdir(dir_path)
    for i in range(len(filename_list)):
        if (i+1) in chosen_file_index:
            each_mat_filepath = os.path.join(dir_path, filename_list[i])
            print(each_mat_filepath, filename_list[i].split('.')[0])
            mat_filepath_list.append(each_mat_filepath)
            mat_name_list.append(filename_list[i].split('.')[0])
    return mat_filepath_list, mat_name_list


def preprocess_fc_matrix(matrix_filepath, ratio, save_dir, filename):
    # load matrix
    arr = scio.loadmat(matrix_filepath)['DRStruct'][0][0][0]
    # preprocess
    baseline = np.quantile(abs(arr), 1-ratio)
    arr[arr < baseline] = 0
    arr_sparse = sparse.csr_matrix(arr)
    graph = dgl.from_scipy(arr_sparse)
    # generate_dgl_graph
    min_max_scaler = MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(arr)
    graph.ndata['feat'] = torch.DoubleTensor(scaled_array)
    # save_dgl_graph
    print(filename + ' preprocess finish')
    save_graphs(os.path.join(save_dir, filename+'.pkl'), graph)


def batch_load_preprocess(dir_path, chosen_file_index, ratio, save_dir):
    mat_filepath_list, mat_name_list = load_mat_filepath_from_dir(dir_path, chosen_file_index)
    for i in range(len(mat_name_list)):
        each_mat_filepath = mat_filepath_list[i]
        filename = mat_name_list[i]
        preprocess_fc_matrix(each_mat_filepath, ratio, save_dir, filename)


if __name__ == '__main__':
    save_dir = 'D:\\PyCharmProjects\\gnn_pd\\data\\graph'
    dir_path = "E:\\fMRI_PD\\DFC"
    ratio = 0.15
    batch_load_preprocess(dir_path, chosen_file_index, ratio, save_dir)