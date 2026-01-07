from turtle import pd
import torch
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random

import pickle
def read_pickle(filename):
    try:
        with open(filename,'rb') as f:
            data = pickle.load(f)
    except:
        with open(filename,'rb') as f:
            data = pickle.load(f,encoding='latin1')
    return data

class MyDataset(Dataset):

    def __init__(self, dataset_name = 'IEMOCAP', split = 'train', args = None):

        #来自DAG的初始化


        self.args = args
        self.dataset_name = dataset_name
        self.speaker_vocab, self.label_vocab = self.load_vocab()
        self.data = self.read(split)

        print(len(self.data))

        self.len = len(self.data)

    def load_vocab(self):
        speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (self.dataset_name), 'rb'))
        label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % (self.dataset_name), 'rb'))

        return speaker_vocab, label_vocab



    def read(self, split):


        #来自DAG的读取
        with open('./data/%s/%s_data_roberta.json.feature'%(self.dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)

        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            for i,u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers':speakers,
                'features': features
            })

        random.shuffle(dialogs) # 打乱对话
        return dialogs

    def __getitem__(self, index):  # 获取一个样本/ 对话
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''

        #来自DAG
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

    def get_all_reps_and_labels(self): #获取所有数据集的表征和标签用于执照支持数据集
        reps = []
        labels = []
        for i, d in enumerate(self.data):
            reps.append(torch.FloatTensor(d['features']))
            labels.append(torch.LongTensor(d['labels']))
        return reps, labels


    def __len__(self):
        return self.len
    
    def get_semantic_adj(self, speakers, max_dialog_len):
  
        semantic_adj = []
        for speaker in speakers:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long) # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)): # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        if i==j:
                            s[i,j] = 1  # 对角线  self
                        elif i < j:
                            s[i,j] = 2   # self-future
                        else:
                            s[i,j] =3    # self-past
                    else:
                        if i<j:
                            s[i,j] = 4   # inter-future
                        elif i>j:
                            s[i,j] = 5   # inter-past
                        

            semantic_adj.append(s)
        
        return torch.stack(semantic_adj)


    def get_structure_adj(self, links, relations, lengths, max_dialog_len):
        '''
        map_relations = {'Comment': 0, 'Contrast': 1, 'Correction': 2, 'Question-answer_pair': 3, 'QAP': 3, 'Parallel': 4, 'Acknowledgement': 5,
                     'Elaboration': 6, 'Clarification_question': 7, 'Conditional': 8, 'Continuation': 9, 'Result': 10, 'Explanation': 11,
                     'Q-Elab': 12, 'Alternation': 13, 'Narration': 14, 'Background': 15}

        '''
        structure_adj = []

        for link,relation,length in zip(links,relations,lengths):  
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long) # （N,N） 0 表示填充部分 或 没有关系
            assert len(link)==len(relation)

            for index, (i,j) in enumerate(link):
                s[i,j] = relation[index] + 1
                s[j,i] = s[i,j]   # 变成对称矩阵了

            for i in range(length):  # 填充对角线
                s[i,i] = 17

            structure_adj.append(s)
        
        return torch.stack(structure_adj)




    ## 来自DAG-ERC的有向无环图标注方法
    def get_adj_local(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                scnt = 0
                for j in range(i - 1, -1, -1):
                    # if speaker[j] == s and scnt <= 1:
                    if speaker[j] == s:
                        a[j,i] = 1
                        a[i,j] = 3
                        scnt += 1
                        cnt += 1
                    if speaker[j] != s and cnt < self.args.local_window:
                        a[j,i] = 2
                        a[i,j] = 4
                        cnt += 1
            adj.append(a)
        return torch.stack(adj)

    ## 来自MuCDN的图标注方法
    def get_adj_global(self, speakers, max_dialog_len):

        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s:
                        a[j, i] = 1
                        a[i, j] = 3
                    if speaker[j] != s and cnt < self.args.global_window:
                        a[j, i] = 2
                        a[i, j] = 4
                        cnt += 1
            adj.append(a)
        return torch.stack(adj)


    def collate_fn(self, data):  # data 是一个batch 的对话    获取一批样本/对话 并填充

        '''
        :param data:
            utterance_features, labels, speakers, utterance_links, utterance_relations, length, texts,id
        :return:
            text_features: (B, N, D) padded
    
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker information for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''

        # 来自DAG
        '''
        data:
            feature,
            label
            speaker
            length
            text
        '''
        max_dialog_len = max([d[3] for d in data])  # batch 中 对话的最大长度 N
        utterance_features = pad_sequence([d[0] for d in data], batch_first = True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value = -1) # (B, N ) label 填充值为 -1

        semantic_adj = self.get_semantic_adj([d[2] for d in data], max_dialog_len)
        # semantic_adj = self.get_adj_global([d[2] for d in data], max_dialog_len)
        structure_adj = self.get_adj_local([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first = True, padding_value = -1)  # (B, N) speaker 填充值为 -1
        utterances = [d[4] for d in data]  # batch 中每个对话对应的 utterance 文本
        return utterance_features, labels, semantic_adj, structure_adj, lengths, speakers, utterances