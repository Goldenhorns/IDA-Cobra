from importlib.resources import path
import numpy as np
import scipy.sparse as sp
import random
import scipy.io as sio
import argparse
import pickle as pkl
import networkx as nx
import sys
import os
import torch
from tqdm import tqdm
import os.path as osp
from sklearn import preprocessing
from scipy.spatial.distance import euclidean



#region—Fuction
def dense_to_sparse(dense_matrix):
    shape = dense_matrix.shape
    row = []
    col = []
    data = []
    for i, r in enumerate(dense_matrix):
        for j in np.where(r > 0)[0]:
            row.append(i)
            col.append(j)
            data.append(dense_matrix[i,j])

    sparse_matrix = sp.coo_matrix((data, (row, col)), shape=shape).tocsc()
    return sparse_matrix

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_citation_datadet(dataset_str,path):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}\{}\ind.{}.{}".format(path,dataset_str,dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}\{}\ind.{}.test.index".format(path,dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj_dense = np.array(adj.todense(), dtype=np.float64)
    attribute_dense = np.array(features.todense(), dtype=np.float64)
    cat_labels = np.array(np.argmax(labels, axis = 1).reshape(-1,1), dtype=np.uint8)

    return attribute_dense, adj_dense, cat_labels
#endregion

#region-Set Argument
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r"C:\Users\17855\Desktop\dataset")
parser.add_argument('--dataset', type=str, default='cora')  #'BlogCatalog'  'Flickr' 'cora'  'citeseer'  'pubmed' 
parser.add_argument('--seed', type=int, default=1)#random seed
parser.add_argument('--m', type=int, default=15)  #num of fully connected nodes
parser.add_argument('--k', type=int, default=50) 
parser.add_argument('--n', type=int)              #num of clusters
args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
AD_dataset_list = ['BlogCatalog', 'Flickr']
Citation_dataset_list = ['cora', 'citeseer', 'pubmed']
#endregion

#region-Set hyperparameters of disturbing
dataset_str = args.dataset
path=args.path
seed = args.seed
m = args.m
k = args.k
if args.n is None:
    if dataset_str == 'cora' or dataset_str == 'citeseer':
        n = 5
    elif dataset_str == 'BlogCatalog':
        n = 10
    elif dataset_str == 'Flickr':
        n = 15
    elif dataset_str == 'pubmed':
        n = 20
else:
    n = args.n
#endregion

if __name__ == "__main__":

    #Set seed
    print('Random seed: {:d}. \n'.format(seed))
    np.random.seed(seed)
    random.seed(seed)

    #region-Load data
    print('Loading data: {}...'.format(dataset_str))
    if dataset_str in AD_dataset_list:
        data = sio.loadmat('{}/{}/{}.mat'.format(path,dataset_str, dataset_str))
        attribute_dense = np.array(data['Attributes'].todense())
        attribute_dense = preprocessing.normalize(attribute_dense, axis=0)
        adj_dense = np.array(data['Network'].todense())
        cat_labels = data['Label']
    elif dataset_str in Citation_dataset_list:
        attribute_dense, adj_dense, cat_labels = load_citation_datadet(dataset_str,path)
    #endregion

    ori_num_edge = np.sum(adj_dense) 
    num_node = adj_dense.shape[0]    

    #region-Random pick anomaly nodes
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    anomaly_idx = all_idx[:m*n*2]
    structure_anomaly_idx = anomaly_idx[:m*n]
    attribute_anomaly_idx = anomaly_idx[m*n:]
    label = np.zeros((num_node,1),dtype=np.uint8)
    label[anomaly_idx,0] = 1
    

    str_anomaly_label = np.zeros((num_node,1),dtype=np.uint8)
    str_anomaly_label[structure_anomaly_idx,0] = 1
    attr_anomaly_label = np.zeros((num_node,1),dtype=np.uint8)
    attr_anomaly_label[attribute_anomaly_idx,0] = 1
    edge_label=np.zeros([num_node,num_node])
    #endregion

    #region-Disturb structure
    print('Constructing structured anomaly nodes...')
    pbar0=tqdm(range(n),colour="BLUE")
    for n_ in pbar0 :
        pbar0.set_description('Disturb structure')
        current_nodes = structure_anomaly_idx[n_*m:(n_+1)*m]
        print(current_nodes)
        for i in current_nodes:
            for j in current_nodes:
                if adj_dense[i, j]==0:
                    edge_label[i,j]=1
                adj_dense[i, j] = 1.
        edge_label[current_nodes,current_nodes]=0
        adj_dense[current_nodes,current_nodes]= 0.
    real_add_edge=np.sum(edge_label)
    num_add_edge =m*(m-1)*n
    print('Done. {:d} structured nodes are constructed. ({:.0f} edges are added , unwangted edge:{} ) \n'.format(len(structure_anomaly_idx),num_add_edge,real_add_edge))
    #endregion

    #region-Disturb attribute
    print('Constructing attributed anomaly nodes...')
    pbar=tqdm(attribute_anomaly_idx,colour="BLUE")
    for i_ in pbar:
        pbar.set_description('Disturb attribute')
        picked_list = random.sample(all_idx, k)
        max_dist = 0
        for j_ in picked_list:
            cur_dist = euclidean(attribute_dense[i_],attribute_dense[j_])
            if cur_dist > max_dist:
                max_dist = cur_dist
                max_idx = j_
        graph = nx.from_numpy_matrix(adj_dense)
        neibor1=[]
        neibor2=[]
        for q in graph.neighbors(i_):
            neibor1.append(q)
        for w in graph.neighbors(max_idx):
            neibor2.append(w)
        for e in neibor2:
            if e not in neibor1:
                edge_label[max_idx,e]=1
                edge_label[e,max_idx]=1
        for r in neibor1:
            if r not in neibor2:
                edge_label[i_,r]=1
                edge_label[r,i_]=1                
        attribute_dense[i_] = attribute_dense[max_idx]
    add_edge=np.sum(edge_label)-real_add_edge
    print('Done. {:d} attributed nodes are constructed,unwanted edge{}. \n'.format(len(attribute_anomaly_idx),add_edge))
    #endregion
    
    #region-Pack & save them into .mat
    print('Saving mat file...')
    attribute = dense_to_sparse(attribute_dense)
    adj = dense_to_sparse(adj_dense)
    edge_label=dense_to_sparse(edge_label)
    savedir = r"C:\Users\17855\Desktop\dataset"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    sio.savemat(r"C:\Users\17855\Desktop\data\{}.mat".format(dataset_str),\
                {'Network': adj, 'Label': label, 'Attributes': attribute,\
                'Class':cat_labels, 'str_anomaly_label':str_anomaly_label,\
                'attr_anomaly_label':attr_anomaly_label,'edge_label':edge_label})
    #endregion
    
    print('Done. The file is save as: {}.mat \n'.format(dataset_str))


