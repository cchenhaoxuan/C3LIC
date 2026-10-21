import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
from utils import label2onehot, Quantization_S, calc_simY,calc_train_label,eval_turn
import clip
import re
import matplotlib.pyplot as plt

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

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1)  #行和1
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)  #列和N/K
    return Z + u.unsqueeze(1) + v.unsqueeze(0)
def log_optimal_transport(scores: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    m, n = scores.shape #n->K, m->N
    one = scores.new_tensor(1)
    ms, ns = ((m/n)*one).to(scores), (one).to(scores) #
    couplings = scores
    log_mu = ns.log()[None].expand(m)
    log_nu = ms.log()[None].expand(n)
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    # Z = Z - norm  # multiply probabilities by M+N
    return Z



def train_model(model, dataloader, optimizer,scheduler, num_epochs, bits,log_file,theta,tao):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_quan_S = Quantization_S(subLevel=bits // 8, subCenters=256, dim=256,theta=theta,tao=tao)
    model_quan_S = model_quan_S.cuda()
    clip_model = '/home/dell/data/code/whb/pretrained/ViT-B-16.pt'
    device = 'cuda'
    encoder, _ = clip.load(clip_model, device=device)
    classname_path = '/home/dell/data/dataset/CUB_200_2011/classes.txt'
    classname_list = []

    with open(classname_path, 'r') as f:
        for line in f:
            cls_name = line.strip('\n').split('.')[1]
            cls_name = cls_name.replace('_', ' ')
            classname_list.append(cls_name)
    optimizerQ = optim.Adam(model_quan_S.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    train_labels = label2onehot(torch.tensor(dataloader['train'].dataset.labels),
                                classes=max(dataloader['train'].dataset.labels) + 1)
    train_labels_onehot = label2onehot(torch.tensor(dataloader['train'].dataset.labels),
                                       classes=max(dataloader['train'].dataset.labels) + 1)

    num_train = len(dataloader['train'].dataset.labels)
    Sim = calc_simY(train_labels)
    train_labels_onehot = calc_train_label(train_labels_onehot)
    U = torch.zeros((num_train, 256), dtype=torch.float32)
    weights =  torch.ones((bits // 8 , 1), dtype=torch.float32) #/ 8
    train_codes = np.random.randint(0, 255, size=(num_train, bits // 8), dtype=np.int32)

    fg_text_features = zeroshot_classifier(classname_list, ['a clean origami of {}.'], encoder)
    fg_text_features = fg_text_features.float()

    for epoch in range(num_epochs):

        model.train()
        ce_loss = 0.0

        for batch_cnt, (inputs, labels, item) in enumerate(dataloader['train']):


            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            optimizerQ.zero_grad()
            logits, q_codes, loss, text_emd = model(inputs,fg_text_features,labels)
            U[item, :] = q_codes.cpu().data

            Distortion = model_quan_S(q_codes, text_emd, torch.from_numpy(train_codes).long().to(inputs.device),
                                      Sim[item.cpu(), :].to(inputs.device), item.to(inputs.device),labels,U,weights.to(inputs.device).detach())

            loss = loss.mean() + Distortion.squeeze()
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizerQ.step()
            ce_loss += loss.item() * inputs.size(0)



        epoch_loss = ce_loss / dataloader['train'].total_item_len
        scheduler.step()


        for level in range(bits // 8):
            codewords = model_quan_S.Codebook[level].cpu()
            codes_hat = U.mm(codewords.t())
            sel_ind = np.setdiff1d([ii for ii in range(bits // 8)], level)
            QHard = 0.
            C_B = 0.
            weights_sum_d = weights[level:bits // 8].sum()
            for jj in sel_ind:
                codes_tild = model_quan_S.Codebook[jj].cpu()
                C_B += codes_tild[train_codes[:, jj], :]
                if jj < level:
                    QHard += weights_sum_d * U.mm(codes_tild[train_codes[:, jj]].t())
                else:
                    QHard += weights[jj:bits // 8].sum() * U.mm(codes_tild[train_codes[:, jj]].t())
            G = (torch.sum(torch.tensor([ii * weights[ii-1] for ii in range(level + 1, bits // 8 + 1)])) * Sim - QHard)
            diff = 2 * G.t().mm(codes_hat) * weights_sum_d
            diff = diff / U.size(-1) ** .5
            Q = log_optimal_transport(diff, 20).transpose(0, 1)
            codes = torch.argmax(Q, dim=0)
            train_codes[:, level] = codes

        Q_err_i = torch.zeros((bits // 8, 1), dtype=torch.float32)
        QHard = 0.
        for level in range(bits // 8):
            codes_tild = model_quan_S.Codebook[level].cpu()
            QHard      = QHard + U.mm(codes_tild[train_codes[:, level]].t())
            Distortion = torch.mean(torch.mean((QHard - (level + 1) * Sim).pow(2), 1))
            Q_err_i[level] = torch.exp(Distortion)
        weights = Q_err_i/Q_err_i.max()


        if (epoch+1) %10==0:
            test_map_64, test_map_48, test_map_32, test_map_16, test_acc = eval_turn(model,
                                                                                     dataloader['val'], train_labels_onehot,
                                                                                     device, train_codes,
                                                                                     model_quan_S.Codebook.cpu())


            avg_usage, level_usages = compute_codebook_usage(train_codes, bits)
            print(f"Epoch {epoch + 1}: Codebook Usage = {avg_usage:.1f}%")
            log_file.write(f"Epoch {epoch + 1}: Codebook Usage = {avg_usage:.1f}%\n")

            print('epoch:{}:  loss:{:.4f},  64bit MAP:{:.6f},48bit MAP:{:.6f},32bit MAP:{:.6f},16bit MAP:{:.6f},ACC:{:.4f}'.format(epoch+1,
                                                                epoch_loss, test_map_64,test_map_48,test_map_32,test_map_16,test_acc))
            log_file.write('epoch:{}:  loss:{:.4f},  64bit MAP:{:.6f},48bit MAP:{:.6f},32bit MAP:{:.6f},16bit MAP:{:.6f},ACC:{:.4f}'.format(epoch+1,
                                                                epoch_loss, test_map_64,test_map_48,test_map_32,test_map_16,test_acc) + '\n')
