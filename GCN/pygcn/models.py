import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    """
    The Source is "https://github.com/tkipf/pygcn"
    """
	
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
		# This gcn consists of two GraphConvolution layers. 
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
		# If you want to increase your layer.
		# Use like this.(Final layer output size is nclass!)
		# self.gc3 = GraphConvolution(nhid, nhid)
		# self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
		# If you want to increase your layer.
		#x = self.gc3(x, adj)
		#x = F.dropout(x, self.dropout, training=self.training)
		#x = self.gc4(x, adj)
		#x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
