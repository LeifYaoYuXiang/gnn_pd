import os, random
from dgl.data.utils import load_graphs
from utils import read_list_from_file


class Dataloader():
    def __init__(self, data_dir, data_type, label_filepath):
        data_type_dir = os.path.join(data_dir)
        self.graph_list = []
        for each_graph in os.listdir(data_type_dir):
            each_graph_path = os.path.join(data_type_dir, each_graph)
            print(each_graph_path)
            self.graph_list.append(load_graphs(str(each_graph_path))[0][0])
        self.label_list = list(map(float, read_list_from_file(label_filepath)))
        self.data_list = []
        assert len(self.graph_list) == len(self.label_list)
        length = len(self.graph_list)
        for i in range(length):
            self.data_list.append((self.graph_list[i], self.label_list[i]))

    def generate_dataloader(self, train_test_ratio):
        random.shuffle(self.data_list)
        train_index = int(len(self.data_list) * train_test_ratio)
        train_dataloader = self.data_list[0: train_index]
        test_dataloader = self.data_list[train_index+1: len(self.data_list)]
        return train_dataloader, test_dataloader
