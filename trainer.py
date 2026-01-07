import numpy as np
import torch

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report


import json


def train_or_eval_model(model, loss_function, dataloader, device, args, optimizer=None, train=False):
    losses, preds, labels = [], [], []

    emolosses, difflosses, labellosses = [], [], []

    cluster_plot_data = []
    feature_for_plot_all = []
    labe_for_plot_all = []

    # For error analysis: store detailed information
    all_logits = []  # Store all logits (probabilities)
    all_utterances = []  # Store all utterance texts
    all_speakers = []  # Store all speakers
    all_lengths = []  # Store dialogue lengths
    dialog_idx = 0  # Track dialogue index

    assert not train or optimizer != None
    if train:  # 训练模式
        model.train()
    else: # 验证模式
        model.eval()

    for data in dataloader: # 遍历每个batch
        if train:
            optimizer.zero_grad()

        # utterance_features, label, semantic_adj, structure_adj, lengths, speakers, utterances, ids = data
        utterance_features, label, semantic_adj, structure_adj, lengths, speakers, utterances = data


        utterance_features = utterance_features.to(device)
        label = label.to(device)  # (B,N)
        semantic_adj = semantic_adj.to(device)
        structure_adj = structure_adj.to(device)

        log_prob, diff_loss, label_loss, feature_for_plot = model(utterance_features, label, semantic_adj, structure_adj) # (B, N, C)

        # Store detailed information for error analysis (only in eval mode)
        if not train:
            all_logits.append(log_prob.detach().cpu())  # (B, N, C)
            all_utterances.extend(utterances)  # List of dialogue utterances
            all_speakers.append(speakers.cpu())  # (B, N)
            all_lengths.append(lengths.cpu())  # (B,)

        # print('log_prob: ',log_prob, log_prob.size())
        # print('label: ', label.size())
        loss = loss_function(log_prob.permute(0,2,1), label)

        emolosses.append(loss.item())
        difflosses.append(diff_loss.item())
        if args.CL_type != 'None':
            labellosses.append(label_loss.item())

        loss = loss + diff_loss
        if label_loss is not None:
            loss = loss + label_loss

        # 收集特征聚类绘图的特征信息（这里先只用真实标签分类）
        label_for_plot = label.reshape(-1, ).cpu().numpy().tolist()
        feature_for_plot = feature_for_plot.reshape(-1, 300).detach().cpu().numpy().tolist()
        labe_for_plot_all.extend(label_for_plot)
        feature_for_plot_all.extend(feature_for_plot)

        label = label.cpu().numpy().tolist()
        pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist() # (B,N)
        preds += pred
        labels += label
        losses.append(loss.item())


        if train:

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            #检查梯度
            # index = 0
            # for param in model.parameters():
            #     if index<=10:
            #         print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))
            #     index = index+1

    if preds != []:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels): # 遍历每个对话
            for j,l in enumerate(label): # 遍历每个utterance
                if l != -1: # 去除填充标签 （IEMOCAP内部utterance 也有填充）
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    cluster_plot_data.append(feature_for_plot_all)
    cluster_plot_data.append(labe_for_plot_all)

    avg_loss = round(np.sum(losses) / len(losses), 4)

    avg_emoloss = round(np.sum(emolosses) / len(losses), 4)
    avg_diffloss = round(np.sum(difflosses) / len(losses), 4)
    if label_loss is not None:
        avg_labelloss = round(np.sum(labellosses) / len(losses), 4)
    else:
        avg_labelloss = 0

    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP_small', 'EmoryNLP_big', 'EmoryNLP']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    elif args.dataset_name == 'DailyDialog':
        avg_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=[0,2,3,4,5,6]) * 100, 2) #1 is neutral

    fscore_perclass = classification_report(new_labels, new_preds, output_dict=True)

    # Prepare detailed information for error analysis
    detailed_info = None
    if not train and all_logits:
        # Keep as lists - will be processed in error_analysis.py
        # Don't concatenate here because different batches have different max_N
        detailed_info = {
            'logits': all_logits,  # List of tensors (B, max_N, C) for each batch
            'utterances': all_utterances,  # List of lists of utterance texts
            'speakers': all_speakers,  # List of tensors (B, max_N) for each batch
            'lengths': all_lengths,  # List of tensors (B,) for each batch
            'labels': labels,  # Original labels (with padding)
            'preds': preds  # Original predictions (with padding)
        }


    return avg_loss, avg_accuracy, labels, preds, avg_fscore, avg_emoloss, avg_diffloss, avg_labelloss, cluster_plot_data, fscore_perclass, new_labels, new_preds, detailed_info
