import re

import torch
import numpy as np
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist
import clip
import cv2
import os
from sklearn.cluster import KMeans
def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def label2onehot(labels, classes):
    labels_onehot = torch.zeros(labels.size()[0],classes)
    index = labels.view(-1,1)
    labels_onehot.scatter_(dim=1, index = index, value = 1)
    return labels_onehot


def calc_hammingDist(B1, B2):
    # q = B2.shape[1]
    # if len(B1.shape) < 2:
    #     B1 = torch.tensor(B1)
    #     B1 = B1.unsqueeze(0)
    #     B1 = B1.numpy()
    # distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    B1 = B1 / np.linalg.norm(B1)
    B2 = B2 / np.linalg.norm(B2, axis=1)[:, np.newaxis]
    # q = B2.shape[1]
    # distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    # distH = np.sum((B1[np.newaxis,:] - B2) ** 2, 1)
    distH = -np.dot(B1, B2.transpose())
    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):

    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.int)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd, as_tuple=False)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map

def calc_map(ASQDist,queryL, retrievalL,):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}


    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd).astype(np.int64)
        if tsum == 0:
            continue
        # hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(-ASQDist[iter, :])
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        # print(tindex)
        map_ = np.mean(count / (tindex))
        map = map + map_

    return map


def eval_turn(model, data_loader,train_labels_onehot,device,B,C):
    model.eval()
    classname_path = '/home/dell/data/dataset/CUB_200_2011/classes.txt'
    classname_list = []
    with open(classname_path, 'r') as f:
        for line in f:
            cls_name = line.strip('\n').split('.')[1]
            cls_name = cls_name.replace('_', ' ')
            classname_list.append(cls_name)
    with torch.no_grad():
        clip_model = '/home/dell/data/code/whb/pretrained/ViT-B-16.pt'
        device = 'cuda'
        encoder, _ = clip.load(clip_model, device=device)
    with torch.no_grad():
        total = 0
        map_64 = 0
        map_48 = 0
        map_32 = 0
        map_16 = 0
        correct_com = 0
        train_codes = torch.from_numpy(B).long()

        fg_text_features = zeroshot_classifier(classname_list, ['a clean origami {}.'], encoder)
        fg_text_features = fg_text_features.float()

        for step, batch in enumerate(data_loader):  # epoch_iterator
            batch = tuple(t.to(device) for t in batch)
            x, y, item = batch
            inputs = x.cuda()
            labels = y.cuda()

            Q_label = label2onehot(labels.cpu().data, classes=max(data_loader.dataset.labels) + 1)
            logits, q_codes = model(inputs,fg_text_features)
            D = len(train_codes)
            N = q_codes.size()[0]

            QHard = torch.zeros([N, D], dtype=torch.float32)
            QHard_16bit = torch.zeros([N, D], dtype=torch.float32)
            QHard_32bit = torch.zeros([N, D], dtype=torch.float32)
            QHard_48bit = torch.zeros([N, D], dtype=torch.float32)
            num = len(C)

            for level in range(num):
                if level == 2:
                    QHard_16bit = QHard
                if level == 4:
                    QHard_32bit = QHard
                if level == 6:
                    QHard_48bit = QHard
                codes = C[level] # [K, d]


                X_C_H = (q_codes).cpu() @ (codes).cpu().t()
                QHard = QHard + X_C_H[:, train_codes[:, level]]  # + X_C_S[:, train_codes[:, level]]

            # ============================================================================
            ap_64bit = calc_map(QHard.numpy(), Q_label.numpy(), train_labels_onehot.numpy())
            ap_16bit = calc_map(QHard_16bit.numpy(), Q_label.numpy(), train_labels_onehot.numpy())
            ap_32bit = calc_map(QHard_32bit.numpy(), Q_label.numpy(), train_labels_onehot.numpy())
            ap_48bit = calc_map(QHard_48bit.numpy(), Q_label.numpy(), train_labels_onehot.numpy())

            total += labels.size(0)
            map_64 += ap_64bit
            map_48 += ap_48bit
            map_32 += ap_32bit
            map_16 += ap_16bit
            _, pred = torch.max(logits.data, 1)
            correct_com += pred.eq(labels.data).cpu().sum()

        test_map_64 = 1. * float(map_64) / total
        test_map_48 = 1. * float(map_48) / total
        test_map_32 = 1. * float(map_32) / total
        test_map_16 = 1. * float(map_16) / total
        test_acc = 100. * float(correct_com) / total

    return test_map_64, test_map_48, test_map_32, test_map_16, test_acc

class Quantization_S(nn.Module):
    def __init__(self, subLevel, subCenters, dim,theta,tao):
        super(Quantization_S, self).__init__()
        self._stackLevel = subLevel  # 8
        self._subCeters = subCenters  # 256
        self._dim = dim  # input feature size
        self.theta = theta
        self.tao = tao
        self.criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(512,256)
        self.Codebook = Parameter(torch.Tensor(self._stackLevel, self._subCeters, self._dim))
        nn.init.xavier_normal_(self.Codebook)

    def forward(self, input, text_emb, B, Sim,item_indx,labels,U,weights):
        Num_samples = Sim.size()[-1]
        batch_size = input.size()[0]
        self.Distortion = 0.
        self.QHard = torch.zeros([batch_size, Num_samples], dtype=torch.float32).to(device=input.device)
        for level in range(self._stackLevel):
            codes = self.Codebook[level]
            self.QHard = self.QHard + input.mm(codes[B[:, level]].t())
            self.Distortion += weights[level] * torch.mean(torch.mean((self.QHard - (level + 1) * Sim).pow(2), 1))
            ground_truth = level * torch.ones(text_emb.size(0)).long().to(device=text_emb.device)
            scores = torch.zeros((text_emb.size(0), self._stackLevel), dtype=torch.float32).to(device=text_emb.device)
            for j in range(self._stackLevel):
                scores[:,j] = torch.diag(text_emb.mm(self.Codebook[level].t()) @ self.Codebook[j] @ text_emb.t()) / self.tao
            self.Distortion = self.Distortion + self.theta * torch.mean(torch.nn.CrossEntropyLoss(reduction='none')(scores , ground_truth))
        return self.Distortion

def calc_simY(database_labels):
    S = (database_labels.mm(database_labels.t())>0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S
def calc_train_label(train_labels_onehot):
    S = train_labels_onehot
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S



if __name__ == '__main__':
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_L = torch.Tensor([[0, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1]])
    retrieval_L = torch.Tensor([[1, 0, 0, 1],
                                [1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 1, 0]])

    map = calc_map_k(qB, rB, query_L, retrieval_L)
    print(map)
