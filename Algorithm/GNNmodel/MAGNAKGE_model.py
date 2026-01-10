import torch.nn as nn
import torch
from Algorithm.MAGNA_KGE.MAGNA_KGConv import MAGNAKGlayer
from dgl import DGLGraph
import numpy as np


class MAGNAKGE(nn.Module):
    def __init__(self,
                 num_layers: int,
                 in_ent_dim: int,
                 in_rel_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 hop_num: int,
                 alpha: float,
                 n_order_adj: int,
                 num_heads: int,
                 #  topk: int,
                 feat_drop: float,
                 attn_drop: float,
                 negative_slope: float,
                 edge_drop: float,
                 input_drop: float,
                #  topk_type: str,
                #  ntriples: int
                 ):
        """
        :param num_layers: number of layers
        :param in_ent_dim: the input dimension of entity
        :param in_rel_dim: the input dimension of relation
        :param topk:
        :param alpha:
        :param hidden_dim:
        :param hop_num:
        :param activation:
        :param feat_drop:
        :param attn_drop:
        :param negative_slope:
        :param residual:
        """
        super(MAGNAKGE, self).__init__()
        self.num_layers = num_layers
        self.trans_layers = nn.ModuleList()
        self.hop_num = hop_num
        # self.top_k = topk
        self.alpha = alpha
        self.edge_drop = edge_drop
        # self.ntriples = ntriples
        self.feat_drop_out = nn.Dropout(feat_drop)

        self.n_order_adj = n_order_adj

        self.trans_layers.append(MAGNAKGlayer(in_ent_feats=in_ent_dim, num_heads=num_heads, in_rel_feats=in_rel_dim, out_feats=hidden_dim, feat_drop=feat_drop, hop_num=self.hop_num,
                                         alpha=self.alpha, attn_drop=attn_drop, negative_slope=negative_slope, input_drop=input_drop))
        for l in range(1, num_layers):
            self.trans_layers.append(MAGNAKGlayer(in_ent_feats=hidden_dim,  num_heads=num_heads, in_rel_feats=in_rel_dim, out_feats=hidden_dim, hop_num=self.hop_num,
                                                       alpha=self.alpha,
                                                       feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, input_drop=input_drop))
        # self.fc_rel = nn.Linear(in_rel_dim, hidden_dim, bias=False)
        print('num of node in each batch:', 2 * n_order_adj * (n_order_adj + 1) + 1)
        self.policy =  nn.Sequential(
            nn.Linear(in_features=hidden_dim * (2 * n_order_adj * (n_order_adj + 1) + 1), out_features=hidden_dim * 20),
            nn.ReLU(),
            nn.Dropout(p=feat_drop),

            nn.Linear(in_features=hidden_dim * 20, out_features=hidden_dim * 5),
            nn.ReLU(),
            nn.Dropout(p=feat_drop),

            nn.Linear(in_features=hidden_dim * 5, out_features=action_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if isinstance(self.policy, nn.Linear):
            nn.init.xavier_normal_(self.policy.weight.data)


    def forward(self, graph: DGLGraph, entity_embedder, rel_embedder, mask_edge_ids=None):
        h = entity_embedder
        h_r = rel_embedder
        # number_triples = self.ntriples
        # drop_edges_ids = self.get_drop_edge_pair_ids(number_triples=number_triples)
        for l in range(self.num_layers):
            h = self.trans_layers[l](graph, h, h_r)
        each_batch_node_num = 2 * self.n_order_adj * (self.n_order_adj + 1) + 1
        batch_size = entity_embedder.shape[0] // each_batch_node_num
        h = h.view(batch_size, -1)
        logits = self.policy(h)
        return logits
