from pickletools import read_bytes1
import pandas as pd
from pandas import DataFrame
from turtle import st
import numpy as np
import os
import sys
import scipy.sparse as sp
import torch
import torch.nn as nn
from model import *
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import dgl
import datetime 
from joblib import Parallel, delayed
import argparse
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange

#region-Set result sheet name
cur_time = datetime.datetime.now()
sheet_day=cur_time.day
sheet_hour=cur_time.hour
sheet_minute=cur_time.minute
sheet_second=cur_time.second
sheetname=str(sheet_day)+str(sheet_hour)+str(sheet_minute)+str(sheet_second)
#writer = SummaryWriter()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#endregion

#region-Set argument
parser = argparse.ArgumentParser(description='xsj')
parser.add_argument('--dataset', type=str, default='citeseer')     
parser.add_argument('--lr', type=float,default=1e-3)            
parser.add_argument('--weight_decay', type=float, default=0.0)  
parser.add_argument('--seed', type=int, default=1)              
parser.add_argument('--embedding_dim', type=int, default=120)   
parser.add_argument('--num_epoch', type=int,default=100)        
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--drop_out', type=float, default=0.0)      
parser.add_argument('--batch_size', type=int, default=300)      
parser.add_argument('--test_batch_size', type=int, default=10000) 
parser.add_argument('--subgraph_size', type=int, default=4)    
parser.add_argument('--readout', type=str, default='avg')       
parser.add_argument('--alpha', type=float, default=0.5)           
parser.add_argument('--beta', type=float, default=0.5)           
parser.add_argument('--auc_test_rounds', type=int, default=256) 
parser.add_argument('--negsamp_ratio', type=int, default=1)     
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = args.batch_size
test_batch_size = args.test_batch_size
subgraph_size = args.subgraph_size
alpha=args.alpha
beta=args.beta
print('Using:',str(device).upper())
print('Dataset:',str(args.dataset).upper())
#endregion
#region-Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#endregion

#region-Load and preprocess data
adj, features, labels, idx_train, idx_val,idx_test, \
ano_label, str_ano_label, attr_ano_label,edge_lable = load_mat(args.dataset)
anomal_sum=sum(ano_label)

raw_adj=adj                                 
features, _ = preprocess_features(features) 
dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

adj = normalize_adj(adj)                    
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
#endregion

#region-Initialize model and optimizer 
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout,args.drop_out)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model.to(device)
features = features.to(device)

adj = adj.to(device)
labels = labels.to(device)

#BCEWithLogitsLoss = Sigmoid+BCELoss
b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
xent = nn.CrossEntropyLoss()

cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1
test_batch_num = nb_nodes // test_batch_size + 1
#endregion

#region-Train model
with tqdm(total=args.num_epoch,colour='BLUE',ncols=100) as pbar:
    pbar.set_description('[ Training ]')
    for epoch in range(args.num_epoch):
        loss_full_batch = torch.zeros((nb_nodes,1)).to(device)
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.

        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        subgraphs=nebadd(subgraphs,raw_adj,subgraph_size)
        subgraphs2 = generate_rwr_subgraph(dgl_graph, subgraph_size)

        for batch_idx in trange(batch_num,desc='[ Batch    ]',leave=False,colour='green',ncols=100):
            optimiser.zero_grad()
            is_final_batch = (batch_idx == (batch_num - 1))
            idx,cur_batch_size=cur_idx(all_idx,batch_size,batch_num,batch_idx)
            lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), \
            torch.zeros(cur_batch_size * args.negsamp_ratio))), 1).to(device)
        
            ba1_train,bf1_train=trace2af_str(idx,subgraphs,adj,features)

            ba2_train,bf2_train=trace2af_con(idx,subgraphs2,adj,features)


            logits1,logits2,suba = model(bf1_train,ba1_train,bf2_train,ba2_train,subgraph_size,"train")
            subdis=torch.pow(suba-ba1_train[:,:subgraph_size,:subgraph_size], 2)
            subdis=torch.mean(subdis)

            loss_all = b_xent(logits1, lbl)
            loss_all2 = b_xent(logits2, lbl)
            

            loss1= torch.mean(loss_all)
            loss2= torch.mean(loss_all2)
            loss=alpha*(beta*loss1+(1-beta)*subdis)+(1-alpha)*loss2

            current_loss = loss.item()

            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

            if not is_final_batch:
                total_loss += loss

        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), sheetname+'best_model.pkl')
        else:
            cnt_wait += 1

        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)
#endregion

#region-Test model
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(sheetname+'best_model.pkl'))
multi_round_str_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_con_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))

with tqdm(total=args.auc_test_rounds,colour='BLUE',ncols=100) as pbar_test:
    pbar_test.set_description('[ Testing  ]')
    for round in range(args.auc_test_rounds):

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        subgraphs1 = generate_rwr_subgraph(dgl_graph, subgraph_size)
        subgraphs2 = generate_rwr_subgraph(dgl_graph, subgraph_size)

        for batch_idx in trange(test_batch_num,desc='[ Batch    ]',leave=False,colour='green',ncols=100):
            optimiser.zero_grad()
            torch.cuda.empty_cache() 

            idx,cur_batch_size=cur_idx(all_idx,test_batch_size,test_batch_num,batch_idx)
            ba1,bf1,neb=C_2_pair_3(idx,subgraphs1,raw_adj,adj,features)
            ba2,bf2=trace2af_con(idx,subgraphs2,adj,features)

            with torch.no_grad():
                ret1,ret2,subadj=model(bf1,ba1,bf2,ba2,subgraph_size,"test")
                subdis=torch.pow(subadj-ba1[:,:subgraph_size,:subgraph_size], 2)
                subdis=torch.mean(subdis,dim=(1,2))
                subdis=torch.sigmoid(subdis)
                logits1 = torch.squeeze(ret1)
            
                logits1 = torch.sigmoid(logits1)

                logits2 = torch.squeeze(ret2)
                logits2 = torch.sigmoid(logits2)


            str_sore_u=torch.zeros(len(subdis)).cuda()
            for i in range(len(idx)):
                A_hati=0
                Log_i=i

                cur_nebnum=len(neb[i])
                A_hat=torch.zeros((cur_nebnum))
                for neb_now in range(cur_nebnum):
                    A_hat[A_hati]=logits1[Log_i]
                    A_hati=A_hati+1
                    Log_i=Log_i+cur_batch_size

                A=torch.ones(cur_nebnum)
                str_sore=torch.dist(A,A_hat, p=2)
                str_sore=torch.sigmoid(str_sore)
                str_sore_u[i]=str_sore

            str_sore_u=beta*str_sore_u+(1-beta)*subdis

            multi_round_str_ano_score[round, idx]=str_sore_u.cpu().numpy()

            ano_score = - (logits2[:cur_batch_size] - logits2[cur_batch_size:]).cpu().numpy()
            multi_round_con_ano_score[round, idx] = abs(ano_score)
            
        pbar_test.update(1)

#endregion

str=np.mean(multi_round_str_ano_score, axis=0)

con=np.mean(multi_round_con_ano_score, axis=0)

ano_score_final=alpha*str-(1-alpha)*con

fpr, tpr, thresholds = metrics.roc_curve(ano_label,ano_score_final , pos_label=1)
precisionkl,reacallkl,k_l=metric(ano_label,ano_score_final,anomal_sum, pos_label=1)
AUC=metrics.auc(fpr, tpr)
print(AUC)

#endregion
