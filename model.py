import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import aggr, GCNConv, global_mean_pool, HypergraphConv, GINConv, global_add_pool
from torch.nn import ModuleList
from torch.nn.parameter import Parameter
import math

softmax = torch.nn.LogSoftmax(dim=1)

class TimeSeriesEncoder(nn.Module):
    def __init__(self, num_rois, time_steps, embedding_size):
        super(TimeSeriesEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=time_steps, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=embedding_size, kernel_size=3, stride=1, padding=1)
        # Removing the linear layer that caused the mismatch
        # If you want to add a linear layer, ensure its input matches the output of conv2
        # self.fc = nn.Linear(num_rois, num_rois)  # Commented out for now

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, time_steps, num_rois) -> (batch_size, num_rois, time_steps)
        x = F.relu(self.conv1(x))  # (batch_size, embedding_size, num_rois)
        x = F.relu(self.conv2(x))  # (batch_size, embedding_size, num_rois)
        x = x.permute(0, 2, 1)  # (batch_size, num_rois, embedding_size)
        return x

# Graph Generator
class GraphGenerator(nn.Module):
    def __init__(self, embedding_size, num_rois):
        super(GraphGenerator, self).__init__()
        self.fc = nn.Linear(embedding_size, num_rois)

    def forward(self, x):
        # print(x.size())
        hA = F.softmax(self.fc(x), dim=-1)  # hA should now be (batch_size, num_rois, num_rois)
        # print(hA.size())
        A = torch.bmm(hA, hA.transpose(1, 2))  # This ensures A has the shape (batch_size, num_rois, num_rois)
        return A

# Graph Predictor using GCN
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# End-to-end model
class FBNetGen(nn.Module):
    def __init__(self, num_rois, time_steps, embedding_size, hidden_channels, num_classes):
        super(FBNetGen, self).__init__()
        self.encoder = TimeSeriesEncoder(num_rois, time_steps, embedding_size)
        self.graph_generator = GraphGenerator(embedding_size, num_rois)
        self.gcn = GCN(embedding_size, hidden_channels, num_classes)

    def forward(self, x, save_graph=False, batch_idx=None):
        x = self.encoder(x)
        # print('x', x.size())
        A = self.graph_generator(x)
        if save_graph and batch_idx is not None:
            torch.save(A, f"saved_graphs/graph_batch_{batch_idx}.pt")
        
        node_features = x.reshape(-1, x.size(2))
        num_nodes = node_features.size(0)
        edge_index = torch.randint(0, num_nodes, (2, 500)).to(x.device)
        batch = torch.repeat_interleave(torch.arange(x.size(0)), x.size(1)).to(x.device)
        output = self.gcn(node_features, edge_index, batch)
        return output

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.1):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.fc = nn.Linear(n_hid, n_class)

    def forward(self, data):
        # 支持 Data / DataBatch
        x, G ,batch = data.x, data.H, data.batch
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.hgc2(x, G))     
        x = global_mean_pool(x, batch)  

        # 分类 logits
        out = self.fc(x)                 
        return F.softmax(out, dim=1)
    

class ResidualHyperGNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels, hidden, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        # 
        num_features = train_dataset[0].num_features
        # Use HypergraphConv for all layers instead of the previous GNN/Cheb/Gin branches.
        # Build a stack of HypergraphConv layers: first maps num_features -> hidden_channels,
        # subsequent layers map hidden_channels -> hidden_channels.
        if num_layers > 0:
            self.convs.append(HypergraphConv(num_features, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(HypergraphConv(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2)+ (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)+ (num_features/2))
            
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )
        
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        xs = [x]
        for conv in self.convs:
            xs.append(conv(xs[-1], hyperedge_index=edge_index, hyperedge_attr=edge_attr).tanh())

        h = []
        upper_tri_indices = torch.triu_indices(x.shape[1], x.shape[1])
        
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t[upper_tri_indices[0], upper_tri_indices[1]] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)

        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return softmax(x)
    
class timeResidualHyperGNNs(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.aggr  = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels

        # ====== 基本信息 ======
        sample       = train_dataset[0]
        num_features = sample.num_features   # x 的特征维度：76
        num_nodes    = sample.num_nodes      # 每个图的节点数：116（ROI 数）
        self.num_nodes    = num_nodes
        self.num_features = num_features

        # ====== HypergraphConv 堆栈 ======
        if num_layers > 0:
            self.convs.append(HypergraphConv(num_features, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(HypergraphConv(hidden_channels, hidden_channels))

        # ====== 维度计算（关键） ======
        # corr 被 reshape 成 [B, R, R]，取上三角：
        corr_vec_dim = num_nodes * (num_nodes + 1) // 2   # 6786

        h_dim        = hidden_channels * num_layers       # 每层 aggr 拼起来
        input_dim1   = corr_vec_dim + h_dim

        # BN for corr 上三角向量
        self.bn  = nn.BatchNorm1d(corr_vec_dim)
        # BN for 多层 GNN 的图级 embedding
        self.bnh = nn.BatchNorm1d(h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, args.num_classes),
        )

    def forward(self, data):

        x, corr, edge_index, batch = data.x, data.corr, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        num_graphs = data.num_graphs
        R = self.num_nodes

        # ====== 1) 时序 x 走 HypergraphConv + aggr → h ======
        xs = [x]
        for conv in self.convs:
            xs.append(conv(xs[-1],
                           hyperedge_index=edge_index,
                           hyperedge_attr=edge_attr).tanh())

        h_list = []
        for xx in xs[1:]:                  # 只用 conv 输出，不用原始 x
            xx_graph = self.aggr(xx, batch)  # [N_nodes, C] -> [B, C]
            h_list.append(xx_graph)

        h = torch.cat(h_list, dim=1)       # [B, hidden_channels * num_layers]
        h = self.bnh(h)

        # ====== 2) corr 变成图级 corr_vec ======
        # corr: [B*R, R] -> [B, R, R]
        corr = corr.reshape(num_graphs, R, R)

        upper_tri_indices = torch.triu_indices(
            R, R, device=corr.device
        )

        # 取每个图的上三角 flatten
        corr_vec = torch.stack([
            t[upper_tri_indices[0], upper_tri_indices[1]] for t in corr
        ])                                  # [B, R*(R+1)/2] = [16, 6786]

        corr_vec = self.bn(corr_vec)

        # ====== 3) 拼接 & 分类 ======
        x_out = torch.cat((corr_vec, h), dim=1)  # [B, 6786 + hidden*num_layers]
        x_out = self.mlp(x_out)
        return softmax(x_out)

    
class ResidualGNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        # 
        num_features = train_dataset[0].num_features
        if args.model=="ChebConv":
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels,K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
        elif args.model=="GINConv":
            mlp = nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GNN(mlp))
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GNN(mlp))
        else:
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2)+ (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)+ (num_features/2))
            
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]


        h = []
        upper_tri_indices = torch.triu_indices(x.shape[1], x.shape[1])
        
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t[upper_tri_indices[0], upper_tri_indices[1]] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)

        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return softmax(x)
    

class HGBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = HypergraphConv(in_ch, out_ch)
        self.mlp  = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch)
        )
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.mlp(x)
        return torch.tanh(x)  



class ResidualHmlpGNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels, hidden, num_layers):
        super().__init__()

        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset[0].num_features
        
        self.convs = ModuleList()
        self.convs.append(HGBlock(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(HGBlock(hidden_channels, hidden_channels))

        input_dim1 = int(((num_features * num_features)/2)+ (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)+ (num_features/2))
            
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )


    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]


        h = []
        upper_tri_indices = torch.triu_indices(x.shape[1], x.shape[1])
        
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t[upper_tri_indices[0], upper_tri_indices[1]] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)

        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return softmax(x)


class ResidualCombineGNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels, hidden, num_layers):
        super().__init__()

        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset[0].num_features

        # --- GCN branch ---
        self.gconvs = ModuleList()
        if num_layers > 0:
            self.gconvs.append(GCNConv(num_features, hidden_channels))
            for _ in range(num_layers - 1):
                self.gconvs.append(GCNConv(hidden_channels, hidden_channels))

        # --- Hypergraph branch ---
        self.hconvs = ModuleList()
        if num_layers > 0:
            self.hconvs.append(HypergraphConv(num_features, hidden_channels))
            for _ in range(num_layers - 1):
                self.hconvs.append(HypergraphConv(hidden_channels, hidden_channels))

        # Total aggregated hidden from both branches (for each intermediate layer)
        total_hidden = hidden_channels * num_layers * 2 if num_layers > 0 else 0

        input_dim1 = int(((num_features * num_features)/2)+ (num_features/2) + total_hidden)
        input_dim = int(((num_features * num_features)/2)+ (num_features/2))
            
        self.bn = nn.BatchNorm1d(input_dim)
        # bnh only created when we have hidden features to normalize
        self.bnh = nn.BatchNorm1d(total_hidden) if total_hidden > 0 else None
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )


    def forward(self, data):

        x, edge_index, hyper_edge_index, hyper_edge_attr, batch = data.x, data.edge_index, data.hyper_edge_index, data.hyper_edge_attr, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # GCN branch feature stack
        xs_g = [x]
        for conv in self.gconvs:
            xs_g += [conv(xs_g[-1], edge_index).tanh()]

        # Hypergraph branch feature stack
        xs_h = [x]
        for conv in self.hconvs:
            xs_h.append(conv(xs_h[-1], hyperedge_index=hyper_edge_index, hyperedge_attr=hyper_edge_attr).tanh())

        h = []
        upper_tri_indices = torch.triu_indices(x.shape[1], x.shape[1])
        x_feat = None

        # take the 0-th entry to form the pairwise upper-tri features (same as other classes)
        for i in range(len(xs_g)):
            if i == 0:
                xx = xs_g[0].reshape(data.num_graphs, x.shape[1], -1)
                x_feat = torch.stack([t[upper_tri_indices[0], upper_tri_indices[1]] for t in xx])
                x_feat = self.bn(x_feat)
            else:
                # aggregate both branches for this layer and append
                agg_g = self.aggr(xs_g[i], batch)
                agg_h = self.aggr(xs_h[i], batch)
                h.append(agg_g)
                h.append(agg_h)

        if len(h) > 0:
            h_cat = torch.cat(h, dim=1)
            if self.bnh is not None:
                h_cat = self.bnh(h_cat)
        else:
            # no hidden layers -> create empty tensor with correct batch size
            h_cat = torch.zeros((x_feat.size(0), 0), device=x_feat.device)

        x_concat = torch.cat((x_feat, h_cat), dim=1)
        x_out = self.mlp(x_concat)
        return softmax(x_out)