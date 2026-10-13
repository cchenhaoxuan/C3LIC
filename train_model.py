import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
from utils import label2onehot, Quantization_S, calc_simY,calc_train_label,eval_turn
import clip
import re
import matplotlib.pyplot as plt
def Calculate_MQ(level, bits, model_quan_S, Pd_1, train_codes, train_labels_onehot,beta):
    diff = 0
    QHard_1 = 0.
    codewords_level = model_quan_S.Codebook[level].cpu()
    for i in range(level, bits // 8):
        codes_hat_Pd = Pd_1[i].mm(codewords_level.t())
        sel_ind_i = np.setdiff1d([ii for ii in range(bits // 8)], level)
        for jj in sel_ind_i:
            QHard_1 += model_quan_S.Codebook[jj][train_codes[:, jj]].cpu().t()
            if jj == level-1:
                Q = (level+1) * train_labels_onehot.t() - Pd_1[i].mm(QHard_1)
                diff += 2 * Q.t().mm(codes_hat_Pd) * beta
            elif jj >= level+1:
                Q = (jj+1) * train_labels_onehot.t() - Pd_1[jj].mm(QHard_1)
                diff += 2 * Q.t().mm(codes_hat_Pd) * beta
        # diff += torch.sum(codes_hat_Pd.pow(2), dim=0).unsqueeze(0) * beta
    return diff


def load_class_descriptions(description_file):
    text_list = []
    with open(description_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^\d+\.(.*?)\. (.+)", line)
            if not match:
                print(f"[未匹配行 {idx+1}] → {line}")  # 记录下没匹配到的行
                continue
            class_name = match.group(1)
            description = match.group(2)
            text_list.append(f"{class_name}, {description}")
    return text_list
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
def compute_codebook_usage(train_codes, bits):

    num_levels = bits // 8
    codebook_size = 256

    level_usages = []
    for level in range(num_levels):
        codes_used = train_codes[:, level]
        unique_used = np.unique(codes_used)
        usage_rate = len(unique_used) / codebook_size * 100
        level_usages.append(usage_rate)

    avg_usage = np.mean(level_usages)
    return avg_usage, level_usages


def log_codebook_usage(train_codes, bits, epoch, log_file=None):
    avg_usage, level_usages = compute_codebook_usage(train_codes, bits)

    print(f"Epoch {epoch}: Codebook Usage = {avg_usage:.1f}%")

    if log_file:
        log_file.write(f"Epoch {epoch}: Codebook Usage = {avg_usage:.1f}%\n")

    return avg_usage

def compute_pairwise_codebook_distances(codebook):
    """
    计算码本中每两层之间的欧氏距离

    Args:
        codebook: 包含8层码本的列表或Parameter，每层形状为 [256, dim] 或 [subCenters, dim]

    Returns:
        distances: 包含28个距离值的列表，每个距离是两层码本之间的平均欧氏距离
        pairwise_info: 包含每对码本索引及其距离的详细信息
    """
    num_levels = len(codebook)  # 应该是8
    distances = []
    pairwise_info = []

    # 计算所有两两组合 (共 C(8,2) = 28 种)
    for i in range(num_levels):
        for j in range(i + 1, num_levels):
            # 获取第i层和第j层的码本
            codebook_i = codebook[i].cpu().detach()  # shape: [256, dim]
            codebook_j = codebook[j].cpu().detach()  # shape: [256, dim]

            # 计算两个码本之间的欧氏距离
            # 方法1: 计算所有对应码字之间的平均距离
            # 由于两层码本没有直接的对应关系，我们计算所有码字对之间的平均距离
            # 或者计算最小匹配距离，这里采用平均距离

            # 计算每对码字之间的欧氏距离矩阵
            # codebook_i: [256, dim], codebook_j: [256, dim]
            # distance_matrix: [256, 256]
            distance_matrix = torch.cdist(codebook_i, codebook_j, p=2)  # 欧氏距离

            # 计算多种距离度量
            avg_distance = distance_matrix.mean().item()  # 平均距离

            distances.append(avg_distance)
            pairwise_info.append({
                'layer_i': i,
                'layer_j': j,
                'avg_euclidean_distance': avg_distance
            })

            print(f"Layer {i} vs Layer {j}: Avg Euclidean Distance = {avg_distance:.6f}")

    return distances, pairwise_info
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

        counts_all = np.zeros(256)
        data1 = np.zeros((bits // 8, 256))
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
                    # QHard += weights[jj:bits // 8].sum() * U.mm(codes_tild[train_codes[:, jj]].t())
                    QHard += weights[jj:bits // 8].sum() * U.mm(codes_tild[train_codes[:, jj]].t())
            G = (torch.sum(torch.tensor([ii * weights[ii-1] for ii in range(level + 1, bits // 8 + 1)])) * Sim - QHard)
            diff = 2 * G.t().mm(codes_hat) * weights_sum_d
            diff = diff / U.size(-1) ** .5
            Q = log_optimal_transport(diff, 20).transpose(0, 1)
            codes = torch.argmax(Q, dim=0)
            train_codes[:, level] = codes
        #     counts_num = np.zeros(256)
        #     unique_elements, counts = np.unique(codes, return_counts=True)
        #     for element, count in zip(unique_elements, counts):
        #         counts_num[element] = count
        #     data1[level] = counts_num
        #     counts_all = counts_num + counts_all
        # indices = np.argsort(counts_all)[::-1]
        # counts_all = counts_all[indices]
        # frequencies_1 = counts_all / (len(codes) * (bits // 8))
        ##Optimize weights throgh Q_err_i
        Q_err_i = torch.zeros((bits // 8, 1), dtype=torch.float32)
        QHard = 0.
        for level in range(bits // 8):
            codes_tild = model_quan_S.Codebook[level].cpu()
            QHard      = QHard + U.mm(codes_tild[train_codes[:, level]].t())
            Distortion = torch.mean(torch.mean((QHard - (level + 1) * Sim).pow(2), 1))
            Q_err_i[level] = torch.exp(Distortion)
        weights = Q_err_i/Q_err_i.max()
        # torch.save(model, "./checkpoints/cub.pth")

        # if (epoch+1) == 90:
        #     np.savetxt('./CUB_allbit_delete.txt', frequencies_1)
        #     plt.plot(frequencies_1, label='OTQ')
        #     plt.xlabel('Codewords')
        #     plt.ylabel('Frequency')
        #     plt.legend(loc='upper right')
        #     plt.show()

        # if (epoch + 1) == 90:
        #     codebook_layers = []
        #     for level in range(bits // 8):
        #         codebook_layers.append(model_quan_S.Codebook[level])
        #
        #     # 计算两两距离
        #     distances, pairwise_info = compute_pairwise_codebook_distances(codebook_layers)
        #
        #     # ✅ 修复：直接指定保存文件名（最安全）
        #     output_file = 'codebook_distances.txt'
        #
        #     with open(output_file, 'w') as f:
        #         f.write("Codebook Pairwise Euclidean Distances\n")
        #         f.write("=" * 60 + "\n")
        #         f.write(f"Total pairs: {len(pairwise_info)}\n")
        #         f.write(f"Number of codebook layers: {bits // 8}\n")
        #         f.write("=" * 60 + "\n\n")
        #
        #         f.write("\n\nSimplified format (only average distances):\n")
        #         f.write("Layer pairs (i,j): distance\n")
        #         for i, dist in enumerate(distances):
        #             layer_i = pairwise_info[i]['layer_i']
        #             layer_j = pairwise_info[i]['layer_j']
        #             f.write(f"({layer_i},{layer_j}): {dist:.8f}\n")
        #
        #     print(f"\nCodebook distances saved to: {output_file}")

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
