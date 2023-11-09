import numpy as np
import networkx as nx
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.io as sio
import random
import dgl
import xlrd
def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat(r"./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    edge_lable=data['edge_label'] if ('edge_label' in data) else 0
    edge_lable=sp.csr_matrix(edge_lable)

    adj = sp.csr_matrix(network)#Compressed Sparse Row matrix
    feat = sp.lil_matrix(attr) 

    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels,edge_lable

def normalize_adj(adj):#D^(-1/2)*A*D^(-1/2)
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):#D^(-1/2)*(A+E)*D^(-1/2)
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
    subv = []
    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)
    return subv


def graph_de(adj):
    nd_num=adj.shape[0]
    adj=adj.toarray()
    graph = nx.from_numpy_matrix(adj)
    dl=[]
    for i in range(nd_num):
        a=0
        for n in graph.neighbors(i):
            a=a+1
        dl.append(a)
    max_d=max(dl)
    min_d=min(dl)
    mean=sum(dl)/nd_num
    return max_d,min_d,mean

def trace2af_con(idx,subgraphs,adj,features):
    ba=[]
    bf=[]
    for i in idx:
        cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
        cur_feat = features[:, subgraphs[i], :]
        ba.append(cur_adj) 
        bf.append(cur_feat)

    ft_size=len(features[0][0])
    subgraph_size=len(subgraphs[0])
    cur_batch_size=len(idx)

    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
    added_adj_zero_col[:, -1, :] = 1.
    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
    if torch.cuda.is_available():
        added_adj_zero_row = added_adj_zero_row.cuda()
        added_adj_zero_col = added_adj_zero_col.cuda()
        added_feat_zero_row = added_feat_zero_row.cuda()

    ba = torch.cat(ba)
    ba = torch.cat((ba, added_adj_zero_row), dim=1)
    ba = torch.cat((ba, added_adj_zero_col), dim=2)
    bf = torch.cat(bf)
    bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)
    if torch.cuda.is_available():
        ba.cuda()
        bf.cuda()
    return ba,bf

def trace2af_str(idx,subgraphs,adj,features):
    ba=[]
    bf=[]
    for i in idx:
        cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
        cur_feat = features[:, subgraphs[i], :]
        ba.append(cur_adj) 
        bf.append(cur_feat)
    
    ba = torch.cat(ba)
    ba[:,-2:,:]=0
    ba[:, :,-2:]=0
    ba[:,-1,-1]=1
    ba[:,-2,-2]=1 
    bf = torch.cat(bf)
    if torch.cuda.is_available():
        ba.cuda()
        bf.cuda()
    return ba,bf

def cur_idx(all_idx,batch_size,batch_num,batch_idx):
    is_final_batch = (batch_idx == (batch_num - 1))
    if not is_final_batch:
        idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    else:
        idx = all_idx[batch_idx * batch_size:]
    cur_batch_size = len(idx)
    return idx,cur_batch_size

def nebadd(subgraphs,adj,subgraph_size):
    adj=adj.toarray()
    all_idx = list(range(len(adj[0])))
    graph = nx.from_numpy_matrix(adj)
    for trace in subgraphs:
        neib=[]
        center=trace[subgraph_size-1]
        neib=list(graph.neighbors(center))
        trace.append(random.choice(neib))
        trace.append(random.choice(all_idx))
    return subgraphs

def C_2_pair_2(idx,subgraphs,rowadj,adj,features):
    rowadj=rowadj.toarray()
    graph = nx.from_numpy_matrix(rowadj)
    
    neb=[]
    neblength=[]
    ba=[]
    bf=[]

    for i in idx:
        cur_neb=[]
        for n in graph.neighbors(i):
            cur_neb.append(n)
            neb.append(n)
        neblength.append(len(cur_neb))    
        cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
        cur_feat = features[:, subgraphs[i], :]
        ba.append(cur_adj) 
        bf.append(cur_feat)
        for i in range(len(cur_neb)-1) :
            ba.append(cur_adj)
            bf.append(cur_feat)

    ft_size=len(features[0][0])
    subgraph_size=len(subgraphs[0])
    cur_batch_size=len(neb)

    added_adj_zero_row = torch.zeros((cur_batch_size, 2, subgraph_size))
    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 2, 2))
    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
    
    if torch.cuda.is_available():
        added_adj_zero_row = added_adj_zero_row.cuda()
        added_adj_zero_col = added_adj_zero_col.cuda()
        added_feat_zero_row = added_feat_zero_row.cuda()
    
    ba = torch.cat(ba)#t([a1],[a2],[a3],...)
    ba = torch.cat((ba, added_adj_zero_row), dim=1)
    ba = torch.cat((ba, added_adj_zero_col), dim=2)
    ba[:,-1,-1]=1
    ba[:,-2,-2]=1

    bf = torch.cat(bf)
    bf2=torch.zeros((len(neb),2,ft_size ))
    if torch.cuda.is_available():
        bf2=bf2.cuda()
    
    for i in range(len(neb)):
        neb1_feat = features[:, neb[i], :]
        neb2_feat = features[:, neb[i], :]
        bf2[i,0]=neb1_feat
        bf2[i,1]=neb2_feat
    
    bf = torch.cat((bf, bf2),dim=1)

    if torch.cuda.is_available():
        ba.cuda()
        bf.cuda()

    return ba,bf,neb,neblength

def C_2_pair_3(idx,subgraphs,rowadj,adj,features):
    rowadj=rowadj.toarray()
    graph = nx.from_numpy_matrix(rowadj)
    neb=[]
    ba=[]
    bf=[]
    maxnum=0
    for i in idx:
        cur_neb=[]
        for n in graph.neighbors(i):
            cur_neb.append(n)

        if len(cur_neb)>20:       
            random.shuffle(cur_neb)
            cur_neb=cur_neb[:20]

        if len(cur_neb) > maxnum:
            maxnum=len(cur_neb)
        neb.append(cur_neb)



    for i in idx:
        cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
        cur_feat = features[:, subgraphs[i], :]
        ba.append(cur_adj) 
        bf.append(cur_feat)

    ft_size=len(features[0][0])
    subgraph_size=len(subgraphs[0])
    cur_batch_size=len(idx)

    added_adj_zero_row = torch.zeros((cur_batch_size, maxnum, subgraph_size))
    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + maxnum, maxnum))
    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
    
    if torch.cuda.is_available():
        added_adj_zero_row = added_adj_zero_row.cuda()
        added_adj_zero_col = added_adj_zero_col.cuda()
        added_feat_zero_row = added_feat_zero_row.cuda()
    
    ba = torch.cat(ba)
    ba = torch.cat((ba, added_adj_zero_row), dim=1)
    ba = torch.cat((ba, added_adj_zero_col), dim=2)
    
    for i in range(1,maxnum+1):
        ba[:,-i,-i]=1

    bf = torch.cat(bf)
    bf2=torch.zeros((cur_batch_size,maxnum,ft_size ))
    if torch.cuda.is_available():
        bf2=bf2.cuda()

    for x1 in range(cur_batch_size):
        for x2 in range(len(neb[x1])):
            bf2[x1,x2] = features[:, neb[x1][x2], :]
    
    bf = torch.cat((bf, bf2),dim=1)
    if torch.cuda.is_available():
        ba.cuda()
        bf.cuda()

    return ba,bf,neb

def sortlog(logits):
    fine_logits=[]
    mid=int(len(logits)/2)
    logits1=logits[0:mid]
    logits2=logits[mid:]
    j=0
    s=0
    for i in range(len(logits)):
        if i%2==0:
            fine_logits.append(logits1[j])
            j=j+1
        else:
            fine_logits.append(logits2[s])
            s=s+1
    return fine_logits

def lable_pre(rowadj,edge_lable,adj_pre):
    pre=[]
    lable=[]
    rowadj=rowadj.toarray()
    edge_lable=edge_lable.toarray()
    adj_pre=adj_pre.numpy()
    for row in range(len(adj_pre[0])):
        for data in adj_pre[row]:
            if data>0:
                pre.append(data)
    for row in range(len(rowadj[0])):
        for col in range(len(rowadj[0])):
            if rowadj[row][col]>0:
                if edge_lable[row][col]>0:
                    lable.append(0)
                if edge_lable[row][col]<=0:
                    lable.append(1)

    return pre,lable

def disctance(rowadj,edge_lable,adj_pre):
    rowadj=rowadj.toarray()
    edge_lable=edge_lable.toarray()
    adj_pre=adj_pre.numpy()
    rowadj=torch.FloatTensor(rowadj)
    edge_lable=torch.FloatTensor(edge_lable)
    adj_pre=torch.FloatTensor(adj_pre)
    if torch.cuda.is_available():
        rowadj= rowadj.cuda()
        edge_lable = edge_lable.cuda()
        adj_pre = adj_pre.cuda()
    ori=abs(rowadj- edge_lable)
    disc=abs(adj_pre-ori)
    print(disc.sum())

def Rocauc(fpr,tpr,AUC,dataset,sheetname):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % AUC)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(r'./{}/Figure/{}.svg'.format(dataset,sheetname),format='svg')



def metric(y_true, y_score,anomal_sum, pos_label=1):
    from sklearn.utils import column_or_1d
    from sklearn.utils.multiclass import type_of_target
    
    y_true_type = type_of_target(y_true)
    if not (y_true_type == "binary"):
        raise ValueError("y_true must be a binary column.")
    
    # Makes this compatible with various array types
    y_true_arr = column_or_1d(y_true)
    y_score_arr = column_or_1d(y_score)
    y_true_arr = y_true_arr == pos_label
    k_l=list(range(50,anomal_sum+50,50))
    reacallkl=[]
    precisionkl=[]
    for k in range(50,anomal_sum+50,50):
        desc_sort_order = np.argsort(y_score_arr)[::-1] 
        y_true_sorted = y_true_arr[desc_sort_order]
        y_score_sorted = y_score_arr[desc_sort_order]
        true_positives = y_true_sorted[:k].sum()
        reacall_atk=true_positives/anomal_sum
        precision_atk=true_positives /k
        reacallkl.append(reacall_atk)
        precisionkl.append(precision_atk)

    return precisionkl,reacallkl,k_l