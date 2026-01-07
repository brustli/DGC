from model_utils import RGAT, DiffLoss, SupProtoConLoss
from SSLC_utils import SSLCL, LabelEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn


class DualGATs(nn.Module):

    def __init__(self, args, centers, num_class, ):
        super().__init__()
        self.args = args
        self.num_class = num_class
        
        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        SpkGAT = []
        DisGAT = []
        # todo 调试是否只在第一层交互 ,需要调查的参数：num_relations包括其填充是否是0
        for i in range(args.gnn_layers):
            SpkGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=6))
            # for _ in range(args.diff_layers):
            #     DisGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=18))
            if i == 0 and args.diff_layers > 1:
                for _ in range(args.diff_layers-1):
                    DisGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=5))
            DisGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=5))

        self.SpkGAT = nn.ModuleList(SpkGAT)
        self.DisGAT = nn.ModuleList(DisGAT)

        #这里是新注册的可训练参数
        self.affine1 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        #初始化自定义参数的初始化
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        self.affine2 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine2.data, gain=1.414)

        self.diff_loss = DiffLoss(args)

        #todo 协调方法的初始化参数
        self.label_loss = SupProtoConLoss(args, centers=centers, num_classes=num_class)

        # todo SSLCL的对比学习方法初始化
        self.SSLCL = SSLCL(2.0, 0.5, 1.0, False,
                           1.0, num_classes = num_class, device = self.args.device)
        self.SSLCL_labelemb = LabelEmbedding(num_class, 2*self.args.hidden_dim, self.args.hidden_dim)
        self.SSLCL_Layers = nn.ModuleList([self.SSLCL_labelemb, self.SSLCL])

        self.beta = args.diff_rating
        self.alpha = args.CL_rating

        #分类层的输入维度
        in_dim = args.hidden_dim *2 + args.emb_dim


        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        # layers += [nn.Linear(args.hidden_dim, num_class)]

        out_layer= [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)
        self.out_layers = nn.Sequential(*out_layer)

        # self.out_for_plot = nn.Sequential(*layers[:-1])

        self.drop = nn.Dropout(args.dropout)

        #对于local图节点的表征进行额外的dropout
        self.local_drop = nn.Dropout(args.local_dropout)

       

    def forward(self, utterance_features, label,semantic_adj, structure_adj):
        '''
        :param tutterance_features: (B, N, emb_dim)
        :param xx_adj: (B, N, N)
        :return:
        '''
        batch_size = utterance_features.size(0)
        H0 = F.relu(self.fc1(utterance_features))  # (B, N, hidden_dim)
        H = [H0]
        diff_loss = 0
        for l in range(self.args.gnn_layers):
            if l==0:
                H1_semantic = self.SpkGAT[l](H[l], semantic_adj)
                H1_structure = self.DisGAT[l](H[l], structure_adj)
                if self.args.diff_layers > 1:    
                    for i in range(self.args.diff_layers-1):
                        # todo 这里dropout可以等循环最后再做
                        H1_structure = self.local_drop(H1_structure)
                        H1_structure = self.DisGAT[i+1](H1_structure, structure_adj)
            else:
                H1_semantic = self.SpkGAT[l](H[2*l-1], semantic_adj)
                # todo 调试是否只在第一层交互
                H1_structure = self.DisGAT[self.args.diff_layers - 1 + l](H[2 * l], structure_adj) # [0 1] 2 3
                # todo !!!!!!!!这里[self.args.diff_layers * l + i]原来是+i应是写错了
                # H1_structure = self.DisGAT[self.args.diff_layers * l](H[2 * l], structure_adj)
                # for i in range(self.args.diff_layers-1):
                #     H1_structure = self.local_drop(H1_structure)
                #     H1_structure = self.DisGAT[self.args.diff_layers*l-1+i](H1_structure, structure_adj)


            diff_loss = diff_loss + self.diff_loss(H1_semantic, H1_structure)
            # BiAffine 

            if self.args.do_CA == 0:

                H1_semantic_out = self.drop(H1_semantic) if l < self.args.gnn_layers - 1 else H1_semantic
                H1_structure_out = self.drop(H1_structure) if l < self.args.gnn_layers - 1 else H1_structure

            else:
                A1 = F.softmax(torch.bmm(torch.matmul(H1_semantic, self.affine1), torch.transpose(H1_structure, 1, 2)),
                               dim=-1)
                A2 = F.softmax(torch.bmm(torch.matmul(H1_structure, self.affine2), torch.transpose(H1_semantic, 1, 2)),
                               dim=-1)

                # 原版的两个节点矩阵对换相乘
                H1_semantic_new = torch.bmm(A1, H1_structure)
                H1_structure_new = torch.bmm(A2, H1_semantic)
                # 改版后只让每个图由自己相乘得出
                # H1_semantic_new = torch.bmm(A1, H1_semantic)
                # H1_structure_new = torch.bmm(A2, H1_structure)

                # 最后一层不需要dropout
                H1_semantic_out = self.drop(H1_semantic_new) if l < self.args.gnn_layers - 1 else H1_semantic_new
                H1_structure_out = self.drop(H1_structure_new) if l < self.args.gnn_layers - 1 else H1_structure_new



            H.append(H1_semantic_out)
            H.append(H1_structure_out)

        H.append(utterance_features) 

        HF = torch.cat([H[-3],H[-2],H[-1]], dim=2)  # (B, N, 2*hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
        # logits = self.out_mlp(HF)

        last_feature = self.out_mlp(HF)

        if self.args.CL_type == "SPCL":
            # 来自SPCL的标签异化损失
            HL = last_feature
            label_loss = self.label_loss(HL, label)
        elif self.args.CL_type == "SSLCL":
            # todo 来自SSLCL的标签损失
            # 这里不需要onehot，直接输入数字的label就可以
            label_for_emb = torch.LongTensor([0, 1,2,3,4,5,6]).to(self.args.device)
            labels_embs = self.SSLCL_Layers[0](label_for_emb)
            label_loss = self.SSLCL_Layers[1](f_t_embs = last_feature, label_embs = labels_embs, labels = label)
        else:
            label_loss = 0.0

        logits = self.out_layers(last_feature)

        # 这里收集用于分类特征对于每种标签的聚类情况
        feature_for_plot = last_feature  # (B, N, 300)

        return logits, self.beta * (diff_loss/self.args.gnn_layers), self.alpha * label_loss, feature_for_plot







