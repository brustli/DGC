import logging
from random import random
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
# import apex

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class DiffLoss(nn.Module):

    def __init__(self, args):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        # input1 (B,N,D)    input2 (B,N,D)

        batch_size = input1.size(0)
        N = input1.size(1)
        input1 = input1.view(batch_size, -1)  # (B,N*D)
        input2 = input2.view(batch_size, -1)  # (B, N*D)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True) # (1,N*D)
        input2_mean = torch.mean(input2, dim=0, keepdims=True) # (1,N*D)
        input1 = input1 - input1_mean     # (B,N*D)
        input2 = input2 - input2_mean     # (B,N*D)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach() # (B,1)
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6) # (B,N*D)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach() # (B,1)
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6) # (B,N*D)


        diff_loss = 1.0/(torch.mean(torch.norm(input1_l2-input2_l2,p=2,dim=1)))
  
        return diff_loss

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, relation = False, num_relation=-1,relation_coding='hard',relation_dim=50):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features # 输入特征维度
        self.out_features = out_features # 输出特征维度
        self.alpha = alpha
        self.concat = concat
        self.relation = relation

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if self.relation:
            if relation_coding=='hard':
                emb_matrix = torch.eye(num_relation)  # num_relation 只有relation=True时 有效
                self.relation_embedding = torch.nn.Embedding.from_pretrained(emb_matrix, freeze = True)  # 每种关系 用one-hot向量表示 且不训练
                self.a = nn.Parameter(torch.empty(size=(2*out_features + num_relation, 1)))
            elif relation_coding=='soft':
                self.relation_embedding = torch.nn.Embedding(num_relation,relation_dim)
                self.a = nn.Parameter(torch.empty(size=(2*out_features + relation_dim, 1)))
        else:
            self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, h, adj):
        # h (B,N,D_in)
        Wh = torch.matmul(h, self.W) # (B, N, D_out)
        
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (B, N, N, 2*D_out)

        if self.relation:
            long_adj = adj.clone().type(torch.LongTensor).cuda()
            relation_one_hot = self.relation_embedding(long_adj)  # 得到每个关系对应的one-hot 固定表示

            #print(relation_one_hot.shape)

            a_input = torch.cat([a_input, relation_one_hot], dim = -1)  # （B, N, N, 2*D_out+num_relation）

        #print(a_input.shape)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # (B, N , N)  所有部分都参与了计算 包括填充和没有关系连接的节点

        zero_vec = -9e15*torch.ones_like(e)  #计算mask ,负无穷
        #print(adj.shape)
        #print(e.shape)
        # TODO: Solve empty graph issue here!
        attention = torch.where(adj > 0, e, zero_vec) # adj中非零位置 对应e的部分 保留，零位置(填充或没有关系连接)置为非常小的负数
        attention = F.softmax(attention, dim=2) # B, N, N
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (B,N,N_out)

        h_prime = self.layer_norm(h_prime)


        if self.concat:
            return F.gelu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        ##Wh (B, N, D_out)

        N = Wh.size()[1] # N
        B = Wh.size()[0] # B
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        #print('Wh', Wh.shape)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1) #(B, N*N, D_out)
        #[1,2,3]>[1,1,2,2,3,3]
        Wh_repeated_alternating = Wh.repeat(1, N, 1) #(B, N*N, D_out)
        #[1,2,3]>[1,2,3,1,2,3]
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)  # (B, N*N, 2*D_out)
        # all_combinations_matrix.shape == (B, N * N, 2 * out_features)

        return all_combinations_matrix.view(B, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class RGAT(nn.Module):
    def __init__(self, args, nfeat, nhid, dropout = 0.2, alpha = 0.2, nheads = 2, num_relation=-1):
        """Dense version of GAT."""
        super(RGAT, self).__init__()
        self.dropout = dropout
    
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, relation = True, num_relation=num_relation) for _ in range(nheads)] # 多头注意力
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False, relation = True, num_relation=num_relation) # 恢复到正常维度
        
        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = LayerNorm(nhid)

    def forward(self, x, adj):
        redisual = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1) # (B,N,num_head*N_out)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.gelu(self.out_att(x, adj))  # (B, N, N_out)
        x = self.fc(x)  # (B, N, N_out)
        x = x + redisual
        x = self.layer_norm(x)
        return x



# todo 入参需要用args去赋值，其他要做默认值，supportset和centers要在进入模型前获取。
class SupProtoConLoss(nn.Module):
    def __init__(self, args, centers, num_classes = 7):
        super().__init__()
        self.args = args
        self.temperature = args.temp
        self.default_centers = centers.squeeze()  # (7, 1024)
        self.pools = {}
        # for idx in range(num_classes):
        #     self.pools[idx] = [self.default_centers[idx]]
        self.num_classes = num_classes
        self.pool_size = args.pool_size
        self.K = args.support_size
        self.eps = 1e-8

        self.fc = nn.Linear(args.emb_dim, args.hidden_dim)


    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + self.eps

    def forward(self, reps, labels, decoupled=False):

        # 支持集转化为合适的对比向量（维度相同）
        default_centers = self.fc(self.default_centers)
        for idx in range(self.num_classes):
            self.pools[idx] = [default_centers[idx]]

        reps = reps.reshape(-1, self.args.hidden_dim)
        labels = labels.reshape(-1)
        # 这里bs是总对话数量 = B*N
        batch_size = reps.shape[0]
        curr_centers = []
        pad_labels = []
        # calculate temporary centers
        for idx in range(self.num_classes):
            if len(self.pools[idx]) >= self.K:
                # if len(self.pools[idx]) > 0:
                tensor_center = torch.stack(self.pools[idx], 0)
                perm = torch.randperm(tensor_center.size(0))
                select_idx = perm[:self.K]
                curr_centers.append(tensor_center[select_idx].mean(0))
                pad_labels.append(idx)
            else:
                curr_centers.append(default_centers[idx])
                pad_labels.append(idx)
        curr_centers = torch.stack(curr_centers, 0)
        pad_labels = torch.LongTensor(pad_labels).to(reps.device)

        concated_reps = torch.cat((reps, curr_centers), 0)  # （B*N+Support, 600）
        concated_labels = torch.cat((labels, pad_labels), 0)
        concated_bsz = batch_size + curr_centers.shape[0]  # （B*N+Support)
        mask1 = concated_labels.unsqueeze(0).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask2 = concated_labels.unsqueeze(1).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask = 1 - torch.eye(concated_bsz).to(reps.device)
        pos_mask = (mask1 == mask2).long()
        rep1 = concated_reps.unsqueeze(0).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        rep2 = concated_reps.unsqueeze(1).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        scores = self.score_func(rep1, rep2)
        scores *= 1 - torch.eye(concated_bsz).to(scores.device)

        scores /= self.temperature
        scores = scores[:batch_size]
        pos_mask = pos_mask[:batch_size]
        mask = mask[:batch_size]
        scores -= torch.max(scores).item()

        scores = torch.exp(scores)
        pos_scores = scores * (pos_mask * mask)
        neg_scores = scores * (1 - pos_mask)
        probs = pos_scores.sum(-1) / (pos_scores.sum(-1) + neg_scores.sum(-1))
        probs /= (pos_mask * mask).sum(-1) + self.eps
        loss = - torch.log(probs + self.eps)
        loss_mask = (loss > 0.3).long()
        loss = (loss * loss_mask).sum() / (loss_mask.sum().item() + self.eps)
        # loss = loss.mean()
        return loss


# todo center返回 这里要写好算法算出聚类中心，把每个类别离中心最近的返回出来
def score_func(x, y):
    return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + 1e-8

def my_support_set(trainset, num_classes=7):

    # todo 在set里面读取全部的表征和标签
    train_data = trainset.data
    reps = []
    labels = []
    for i, d in enumerate(train_data):
        for index, f in enumerate(d['features']):
            reps.append(torch.FloatTensor(f))
            labels.append(d['labels'][index])

    label_space = {}
    label_space_dataid = {}
    centers = []
    for idx in range(num_classes):
        label_space[idx] = []
        label_space_dataid[idx] = []
    for idx, turn_reps in enumerate(reps):
        emotion_label = labels[idx]
        if emotion_label < 0:
            continue
        label_space[emotion_label].append(turn_reps)
        label_space_dataid[emotion_label].append(idx)
    # clustering for each emotion class
    dim = label_space[0][0].shape[-1]

    for emotion_label in range(num_classes):
        x = torch.stack(label_space[emotion_label], 0).reshape(-1, dim)
        num_clusters = 1
        cluster_centers = x.mean(0).unsqueeze(0).cpu()
        logging.info('{} clusters for emotion {}'.format(num_clusters, emotion_label))
        centers.append(cluster_centers)  # 求每个情感类别的中心

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    centers = torch.stack(centers, 0).to(device)  # (7, 1, 1024D)
    print("test")

    #todo 待调试模块
    #根据center寻找每个类别最近的发言表征
    delegate_reps = []
    for emotion_label in range(num_classes):
        temp_center = centers[emotion_label].squeeze()
        temp_label_space = torch.stack(label_space[emotion_label]).to(device)
        best_score = 0
        best_rep = torch.zeros(0)
        for index, rep in enumerate(temp_label_space):
            score = score_func(rep, temp_center)
            if score >= best_score:
                best_score = score
                best_rep = rep
        delegate_reps.append(best_rep.unsqueeze(0))
    delegate_reps = torch.stack(delegate_reps, 0).to(device)  # (7, 1, 1024D)

    return delegate_reps


def get_support_set(trainset, num_classes=7):

    # todo 在set里面读取全部的表征和标签
    train_data = trainset.data
    reps = []
    labels = []
    for i, d in enumerate(train_data):
        for index, f in enumerate(d['features']):
            reps.append(torch.FloatTensor(f))
            labels.append(d['labels'][index])

    label_space = {}
    label_space_dataid = {}
    centers = []
    for idx in range(num_classes):
        label_space[idx] = []
        label_space_dataid[idx] = []
    for idx, turn_reps in enumerate(reps):
        emotion_label = labels[idx]
        if emotion_label < 0:
            continue
        label_space[emotion_label].append(turn_reps)
        label_space_dataid[emotion_label].append(idx)
    # clustering for each emotion class
    dim = label_space[0][0].shape[-1]

    for emotion_label in range(num_classes):
        x = torch.stack(label_space[emotion_label], 0).reshape(-1, dim)
        num_clusters = 1
        cluster_centers = x.mean(0).unsqueeze(0).cpu()
        logging.info('{} clusters for emotion {}'.format(num_clusters, emotion_label))
        centers.append(cluster_centers)  # 求每个情感类别的中心

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    centers = torch.stack(centers, 0).to(device)  # (7C, 1B, 1024D)
    print("test")
    return centers

#todo 这里不一定要用到，因为已经有预训练的向量了
def gen_all_reps(model, data, batch_size = 64):
    model.eval()
    '''
    获取当前模型下所有样本的表示以及对应标签，用这里的输出去做聚类
    '''
    results = []
    label_results = []

    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0  # multiprocessing.cpu_count()
    )
    inner_model = model.module if hasattr(model, 'module') else model
    tq_train = tqdm(total=len(dataloader), position=1)
    tq_train.set_description("generate representations for all data")
    with torch.no_grad():
        for batch_id, batch_data in enumerate(dataloader):
            batch_data = [x.to(inner_model.device()) for x in batch_data]
            sentences = batch_data[0]
            emotion_idxs = batch_data[1]

            outputs = inner_model.gen_f_reps(sentences)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            for idx, label in enumerate(emotion_idxs.reshape(-1)):
                if label < 0:
                    continue
                results.append(outputs[idx])
                label_results.append(label)
            tq_train.update()
    tq_train.close()
    dim = results[0].shape[-1]

    results = torch.stack(results, 0).reshape(-1, dim)
    label_results = torch.stack(label_results, 0).reshape(-1)

    return results, label_results