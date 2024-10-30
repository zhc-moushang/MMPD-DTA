import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from tqdm import tqdm


from torch_geometric.data import Data
class PairData(Data):

    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None,y_s=None,protein=None,name=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.y_s = y_s
        self.protein = protein
        self.name = name
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class TestbedDataset(InMemoryDataset):
    def __init__(self, root=None, dataset=None,pro=None,smiles_dic=None,
                   y=None, transform=None,
                 pre_transform=None,complex_graph=None,ligand_graph=None):


        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process( pro,y,complex_graph,ligand_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass


    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):

        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def process(self,pro, y,complex_graph,ligand_graph=None):
        data_list = []
        for name in tqdm(complex_graph, desc="Processing"):

            label = y[name]
            protein = pro[name]
            c_size, features, edge_index = complex_graph[name]
            c_size1, features1, edge_index1 = ligand_graph[name]
            data = PairData(
                torch.LongTensor(edge_index).transpose(1, 0),
                torch.Tensor(features),
                torch.LongTensor(edge_index1).transpose(1, 0), torch.Tensor(features1),
                torch.FloatTensor([label]),
                torch.LongTensor([protein]),name
                            )
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

