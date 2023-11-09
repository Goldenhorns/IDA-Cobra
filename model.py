import torch
import torch.nn as nn
import torch.nn.functional as F

#region-Readout
class AvgReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class MinReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):

    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 128)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out  
#endregion

#region-GCN
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.line_trans= nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_atr = self.line_trans(seq)
        if sparse:
            adjxatr= torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_atr, 0)), 0)
        else:
            adjxatr= torch.bmm(adj, seq_atr)
        if self.bias is not None:
            adjxatr += self.bias
        return self.act(adjxatr)
#endregion

#region-Discriminator
class Str_Discriminator(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self, n_h, negsamp_round):
        super(Str_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self,candhmv,h_neb,mode):
        scs=[]
        if mode=="train":
            h_pos=h_neb[:,-2,:]
            h_neg=h_neb[:,-1,:]
            
            #positive
            scs.append(self.f_k(h_pos, candhmv))
            # negative
            for _ in range(self.negsamp_round):
                scs.append(self.f_k(h_neg, candhmv))
            logits = torch.cat(tuple(scs))
            
        if mode=="test":
            for i in range(len(h_neb[0,:,0])):
                scs.append(self.f_k(h_neb[:,i,:], candhmv))
            logits = torch.cat(tuple(scs))
        return logits    

class Con_Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Con_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1,:], c_mi[:-1,:]),0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits

class Structure_Decoder(nn.Module):
    def __init__(self,n_h, activation,dropout):
        super(Structure_Decoder, self).__init__()
        self.gc1 = GCN(n_h, n_h,activation)
        self.dropout = dropout
    #Ba,Bf,
    def forward(self, s_n_h, s_str_adj):
        x = F.relu(self.gc1(s_n_h, s_str_adj))
        x = F.dropout(x, self.dropout, training=self.training)
        xT=torch.transpose(x,1,2)
        A_hat=torch.bmm(x, xT, out=None)

        return A_hat
#endregion

#region-Model
class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout,dropout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.Structure = GCN(n_in, n_h, activation)
        self.Context = GCN(n_in, n_h, activation)
        self.Structure_Decoder=Structure_Decoder(n_h,activation,dropout)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc1 = Str_Discriminator(n_h, negsamp_round)
        self.disc2 = Con_Discriminator(n_h, negsamp_round)

    def forward(self, str_attr, str_adj,con_attr,con_adj,subgraph_size,mode,sparse=False):
        h_1 = self.Structure(str_attr,str_adj, sparse)
        Sub_hat=self.Structure_Decoder(h_1[:,:subgraph_size,:],str_adj[:,:subgraph_size,:subgraph_size])
        h_2 = self.Context(con_attr,con_adj, sparse)
        if self.read_mode != 'weighted_sum':

            c1 = self.read(h_1[:,:subgraph_size,:])
            h_mv1 = h_1[:,subgraph_size-1,:]
            
            c2= self.read(h_2[:,: -1,:])
            h_mv2 = h_2[:,-1,:]

        else:
            h_mv1 = h_1[:, -1, :]
            c1 = self.read(h_1[:,: -1,:], h_1[:,-2: -1, :])
            h_mv2 = h_2[:, -1, :]
            c2 = self.read(h_2[:,: -1,:], h_2[:,-2: -1, :])

        a=0.1
        candhmv=a*c1+(1-a)*h_mv1
        h_neb=h_1[:,subgraph_size:,:]
        
        ret2 = self.disc2(c2, h_mv2)
        ret1 = self.disc1(candhmv,h_neb,mode)

        return ret1,ret2,Sub_hat
