import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter,LayerNorm
from sklearn.cluster import KMeans
from math import sqrt


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
EPSILON = np.finfo(np.float32).tiny

class CrossAttention(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads


        # 多头注意力
        self.linear_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feed Forward (MLP)
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, q_text, kv_image):


        B, Nq, _ = q_text.size()
        _, Nk, _ = kv_image.size()

        # ===== Multi-Head Attention =====
        q_proj = self.linear_q(q_text).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nq,dh)
        k_proj = self.linear_k(kv_image).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nk,dh)
        v_proj = self.linear_v(kv_image).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nk,dh)

        attn_score = torch.matmul(q_proj, k_proj.transpose(-1, -2))
        sqrt(self.head_dim)  # (B,H,Nq,Nk)
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)

        attn_out = torch.matmul(attn_prob, v_proj)  # (B,H,Nq,dh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, self.hidden_size)  # (B,Nq,d)
        attn_out = self.attn_out(attn_out)

        x =  self.norm1(q_text + attn_out)

        mlp_out = self.mlp(x)

        out =  self.norm2(x + mlp_out)

        return out  # (B, Nq, d)



class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            dim,
            nhead,
            dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(dim, nhead, proj_drop=dropout)
        self.self_attn_2 = Attention(dim, nhead, proj_drop=dropout)
        self.cross_attn = Attention(dim, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)


        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        x1 = x1.permute(0, 2, 3, 1)
        x1 = x1.view(B, -1, C)
        feature_q, feature_k, feature_v = self.norm1(x1), self.norm1(x1), self.norm1(x1)
        x1 = x1 + self.self_attn(feature_q, feature_k, feature_v)
        x1 = x1 + self.dropout(self.mlp(self.norm1(x1)))

        B, C, H, W = x2.size()
        x2 = x2.permute(0, 2, 3, 1)
        x2 = x2.view(B, -1, C)

        feature_q2, feature_k2, feature_v2 = self.norm2(x2), self.norm2(x2), self.norm2(x2)
        x2_1 = x2 + self.self_attn_2(feature_q2, feature_k2, feature_v2)
        x2_1 = x2_1 + self.dropout(self.mlp(self.norm2(x2_1)))
        feature_q2_1, feature_k2_1, feature_v2_1 = self.norm2(x2_1), self.norm2(x2_1), self.norm2(x2_1)
        x_attn = self.cross_attn(x1, feature_k2_1, feature_v2_1)

        B, N, C = x_attn.size()
        x_attn = x_attn.permute(0, 2, 1).contiguous().view(B, C, 7, 7)
        x_up = F.interpolate(x_attn, size=(14, 14), mode="nearest", antialias=False)
        x_attn = x_up.flatten(2).transpose(1, 2)
        x3 = x2 + x_attn
        x3 = x3 + self.dropout(self.mlp(self.norm2(x3)))
        return x1, x3


def smooth_CE(logits, label, peak):
    # logits - [batch, num_cls]
    # label - [batch]
    batch, num_cls = logits.shape
    label_logits = np.zeros(logits.shape, dtype=np.float32) + (1-peak)/(num_cls-1)
    ind = ([i for i in range(batch)], list(label.data.cpu().numpy()))
    label_logits[ind] = peak
    smooth_label = torch.from_numpy(label_logits).to(logits.device)

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label)
    loss = torch.mean(-torch.sum(ce, -1)) # batch average

    return loss

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class MECOM(nn.Module):
    def __init__(self,drop_pro=0.1):
        super(MECOM, self).__init__()
        self.instance_image = nn.Sequential(
            nn.Linear(1024, 1),
            nn.ReLU())
    def forward(self,image_9):
# Gumbel_Softmax
        score_image = self.instance_image(image_9)
        score_image = score_image.squeeze()
        score_hard_image = F.gumbel_softmax(score_image, dim=-1, hard=True)
        score_hard_image = score_hard_image.unsqueeze(dim=-1)
        image_9 = score_hard_image * image_9
        image_9 = torch.sum(image_9,dim=1,keepdim=True)
        return image_9

class HGNN_1(nn.Module):
    def __init__(self, in_ch, n_out):
        super(HGNN_1, self).__init__()
        self.conv = nn.Linear(in_ch, n_out)
        self.bn = nn.BatchNorm1d(n_out)

    def forward(self, x, G):
        residual = x
        x = self.conv(x)
        x = G.matmul(x)
        x = F.relu(self.bn(x.permute(0,2,1).contiguous())).permute(0,2,1).contiguous() + residual
        return x


class HyperEncoder(nn.Module):
    def __init__(self,channel = [1024, 1024]):
        super(HyperEncoder, self).__init__()


        kernel_size  = 3
        stride = 1
        self.HGNN_layer2 = HyperNet(channel[0], node=14, kernel_size=kernel_size, stride=stride, K_neigs=[1])
        self.HGNN_layer3 = HyperNet(channel[1], node=7, kernel_size=kernel_size, stride=stride, K_neigs=[1])


    def forward(self, x4,x5):
        feature_4 = self.HGNN_layer2(x4)
        feature_5 = self.HGNN_layer3(x5)
        # new_H = self.HGNN_layer2(x)

        return feature_4,feature_5
class HyperNet(nn.Module):
    def __init__(self, channel, node = 28, kernel_size=3, stride=1, K_neigs = None):
        super(HyperNet, self).__init__()
        self.HGNN_layer = HGNN_layer(channel, node = node, kernel_size=kernel_size, stride=stride, K_neigs=K_neigs)

    def forward(self, x):

        b,c,w,h = x.shape
        x = x.view(b,c,-1).permute(0,2,1).contiguous()
        x = self.HGNN_layer(x)
        x = x.permute(0,2,1).contiguous().view(b,c,w,h)
        #
        return x

class HGNN_layer(nn.Module):
    """
        Writen by Shaocong Mo,
        College of Computer Science and Technology, Zhejiang University,
    """

    def __init__(self, in_ch, node = 14, K_neigs=None, kernel_size=5, stride=2):
        super(HGNN_layer, self).__init__()
        self.HGNN = HGNN_1(in_ch, in_ch)
        self.K_neigs = K_neigs

        self.local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        B, N, C = x.shape

        topk_dists, topk_inds, ori_dists, avg_dists = self.batched_knn(x, k=self.K_neigs[0])
        H = self.create_incidence_matrix(topk_dists, topk_inds, avg_dists)
        Dv = torch.sum(H, dim=2, keepdim=True)
        alpha = 1.
        Dv = Dv * alpha
        max_k = int(Dv.max())
        _topk_dists, _topk_inds, _ori_dists, _avg_dists = self.batched_knn(x, k=max_k - 1)
        top_k_matrix = torch.arange(max_k)[None, None, :].repeat(B, N, 1).to(x.device)
        range_matrix = torch.arange(N)[None, :, None].repeat(1, 1, max_k).to(x.device)
        new_topk_inds = torch.where(top_k_matrix >= Dv, range_matrix, _topk_inds).long()
        new_H = self.create_incidence_matrix(_topk_dists, new_topk_inds, _avg_dists)
        # local_H = self.local_H.repeat(B,1,1).to(new_H.device)
        #
        # _H = torch.cat([new_H,local_H],dim=2)
        # _H = torch.cat([new_H, local_H], dim=2)
        _G = self._generate_G_from_H_b(new_H)
        #
        x = self.HGNN(x, _G)

        return x



    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        bs, n_node, n_hyperedge = H.shape


        # the weight of the hyperedge

        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)
        # the degree of the node
        DV = torch.sum(H, dim=2)
        # the degree of the hyperedge

        DE = torch.sum(H, dim=1)


        invDE = torch.diag_embed((torch.pow(DE, -1)))
        DV2 = torch.diag_embed((torch.pow(DV, -0.5)))
        W = torch.diag_embed(W)
        HT = H.transpose(1, 2)



        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:

            G = DV2 @ H @ W @ invDE @ HT @ DV2

            return G


    @torch.no_grad()
    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.
        Args:
            x: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            return x_square + x_inner + x_square.transpose(2, 1)



    @torch.no_grad()
    def batched_knn(self, x, k=1):

        ori_dists = self.pairwise_distance(x)
        avg_dists = ori_dists.mean(-1, keepdim=True)
        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)

        return topk_dists, topk_inds, ori_dists, avg_dists

    @torch.no_grad()
    def create_incidence_matrix(self, top_dists, inds, avg_dists, prob=False):
        B, N, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)
        incidence_matrix = torch.zeros(B, N, N, device=inds.device)

        batch_indices = torch.arange(B)[:, None, None].to(inds.device)  # shape: [B, 1, 1]
        pixel_indices = torch.arange(N)[None, :, None].to(inds.device)  # shape: [1, N, 1]

        incidence_matrix[batch_indices, pixel_indices, inds] = weights

        return incidence_matrix.permute(0,2,1).contiguous()



    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):
        if prob:
            # Chai's weight function
            topk_dists_sq = topk_dists.pow(2)
            normalized_topk_dists_sq = topk_dists_sq / avg_dists
            weights = torch.exp(-normalized_topk_dists_sq)
        else:
            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights

    @torch.no_grad()
    def local_kernel(self, size, kernel_size=3, stride=1):
        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]

        inp_unf = torch.nn.functional.unfold(inp, kernel_size=(kernel_size, kernel_size), stride=stride).squeeze(
            0).transpose(0, 1).long()

        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()

        H_local = torch.zeros((size * size, edge))


        H_local[inp_unf, matrix] = 1.

        return H_local


class SubsetOperator(nn.Module):
    def __init__(self, topk, dim, tau=1.0, eps=1e-10):
        super(SubsetOperator, self).__init__()
        self.k = topk
        self.tau = tau
        self.eps = eps
        self.score_fn = nn.Linear(dim, 1)

    def forward(self, x):
        B, n, D = x.shape
        scores = self.score_fn(x).squeeze(-1)  # (B, n)
        gumbel_noise = -torch.empty_like(scores).exponential_().log()
        scores = scores + gumbel_noise
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for _ in range(self.k):
            mask = (1.0 - onehot_approx).clamp(min=self.eps)
            logits = scores + torch.log(mask)
            onehot_approx = F.softmax(logits / self.tau, dim=1)
            khot = khot + onehot_approx
        _, topk_idx = torch.topk(khot, self.k, dim=1)
        hard_mask = torch.zeros_like(khot).scatter_(1, topk_idx, 1.0)
        res = hard_mask - khot.detach() + khot
        selected_topk = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, k, D)
        return selected_topk

class HGNN(nn.Module):
    def __init__(self, nb_classes, sz_embed, hidden):
        super(HGNN, self).__init__()
        self.theta1 = nn.Linear(sz_embed, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lrelu = nn.LeakyReLU(0.1)
        # self.theta2 = nn.Linear(hidden, nb_classes)
    def compute_G(self, H):
        n_edge = H.size(1)
        we = torch.ones(n_edge).cuda()
        Dv = (H*we).sum(dim=1)
        De = H.sum(dim=0)#torch.pow(10, -6)
        We = torch.diag(we)
        inv_Dv_half = torch.diag(torch.pow(Dv, -0.5))
        inv_De = torch.diag(torch.pow(De, -1))
        H_T = torch.t(H)
        G = torch.chain_matmul(inv_Dv_half, H, We, inv_De, H_T, inv_Dv_half)
        return G

    def forward(self, X, H):
        G = self.compute_G(H)
        X = G.matmul(self.theta1(X))
        X = self.bn1(X)
        X = self.lrelu(X)
        out = G.matmul(X)
        return out
class Clfloss(nn.Module):
    def __init__(self,tau):
        super(Clfloss,self).__init__()
        self.tau = tau
    def forward(self,logit,lables):
        logits = logit / self.tau
        loss = F.cross_entropy(logits,lables)
        return loss
class CrossAttention(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads


        # 多头注意力
        self.linear_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feed Forward (MLP)
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, q_text, kv_image):


        B, Nq, _ = q_text.size()
        _, Nk, _ = kv_image.size()

        # ===== Multi-Head Attention =====
        q_proj = self.linear_q(q_text).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nq,dh)
        k_proj = self.linear_k(kv_image).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nk,dh)
        v_proj = self.linear_v(kv_image).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nk,dh)

        attn_score = torch.matmul(q_proj, k_proj.transpose(-1, -2))
        sqrt(self.head_dim)  # (B,H,Nq,Nk)
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)

        attn_out = torch.matmul(attn_prob, v_proj)  # (B,H,Nq,dh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, self.hidden_size)  # (B,Nq,d)
        attn_out = self.attn_out(attn_out)

        # ===== Residual + Norm =====
        x =  self.norm1(q_text + attn_out)

        # ===== Feed Forward (MLP) =====
        mlp_out = self.mlp(x)

        # ===== Residual + Norm =====
        out =  self.norm2(x + mlp_out)

        return out  # (B, Nq, d)
class ResNet50(nn.Module):
    def __init__(self, classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.num_classes = classes
        self.aggregation = nn.Sequential(GeM(work_with_tokens=None), Flatten())
        self.Conv_1 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.fc_2 = nn.Sequential(
            nn.Linear(1024, classes),
        )
        self.fc_two = nn.Sequential(
            nn.Linear(1024, classes),
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(1024, classes),
        )
        self.fc_three = nn.Sequential(
            nn.Linear(1024, classes),
        )
        self.hnmp_last = HGNN(nb_classes=classes, sz_embed=1024, hidden=1024)

        self.fc_q = nn.Sequential(
            nn.Linear(classes, 256),
        )
        self.fc_logit = nn.Sequential(
            nn.Linear(512, 1),
        )
        self.fc_text = nn.Sequential(
            nn.Linear(1024, 256),
        )
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.model.fc = nn.Linear(2048, classes)
        self.part_norm = LayerNorm(1024, eps=1e-6)
        self.trans_layer = TransformerEncoderLayer(dim=1024,nhead=8,dropout=0.1)
        self.text_proj_1 = nn.Linear(512,1024)
        self.logit_scale = nn.Parameter(torch.ones([])) * torch.log(torch.tensor(1/0.007))
        self.clfloss = Clfloss(tau=0.1)
        self.subset = SubsetOperator(topk=2,dim=1024,tau=1.0,eps=1e-6)
        self.transform = CrossAttention(hidden_size=1024,num_heads=8)

    def forward(self, x, Tc, labels=None):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x_3 = self.model.layer3(x)
        x = self.model.layer4(x_3)

        x_4 = self.Conv_1(x)

        x1, x3 = self.trans_layer(x_4, x_3)
        B, N, C = x1.size()
        x1 = x1.permute(0, 2, 1).view(B, C, 7, 7)
        B, N, C = x3.size()
        x3 = x3.permute(0, 2, 1).view(B, C, 14, 14)


        x_p = x3
        x00 = self.aggregation(x_p)
        x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61 = self.aggregation(
            x_p[:, :, 0:5, 0:7]), self.aggregation(
            x_p[:, :, 5:10, 0:7]), self.aggregation(x_p[:, :, 10:, 0:7]), \
            self.aggregation(x_p[:, :, 0:5, 7:]), self.aggregation(
            x_p[:, :, 5:10, 7:]), self.aggregation(x_p[:, :, 10:, 7:]), self.aggregation(x_p[:, :, 0:7, 0:5]), \
            self.aggregation(x_p[:, :, 7:, 0:5]), self.aggregation(x_p[:, :, 0:7, 5:10]), self.aggregation(
            x_p[:, :, 7:, 5:10]), \
            self.aggregation(x_p[:, :, 0:7, 10:]), self.aggregation(x_p[:, :, 7:, 10:])

        # 1x2
        x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x110, x111, x112, x113 = self.aggregation(
            x_p[:, :, 0:10, 0:7]), self.aggregation(
            x_p[:, :, 5:, 0:7]), self.aggregation(x_p[:, :, 0:10, 7:]), \
            self.aggregation(x_p[:, :, 5:, 7:]), self.aggregation(
            x_p[:, :, 0:5, :]), self.aggregation(x_p[:, :, 5:10, :]), \
            self.aggregation(x_p[:, :, 10:, :]), self.aggregation(
            x_p[:, :, :, 0:5]), self.aggregation(x_p[:, :, :, 5:10]), \
            self.aggregation(x_p[:, :, :, 10:]), self.aggregation(
            x_p[:, :, 0:7, 0:10]), self.aggregation(x_p[:, :, 0:7, 5:]), self.aggregation(
            x_p[:, :, 7:, 0:10]), self.aggregation(x_p[:, :, 7:, 5:])

        x30, x31, x32, x33 = self.aggregation(x_p[:, :, 0:10, :]), self.aggregation(x_p[:, :, 5:, :]), \
            self.aggregation(x_p[:, :, :, 0:10]), self.aggregation(x_p[:, :, :, 5:])
        # 1x3
        x40, x41, x42, x43 = self.aggregation(x_p[:, :, :, 0:7]), self.aggregation(x_p[:, :, :, 7:]), \
            self.aggregation(x_p[:, :, 0:7, :]), self.aggregation(x_p[:, :, 7:, :])

        x_mul_v1 = [i.unsqueeze(1) for i in [x50, x51, x52, x53, x54, x55]]
        x_mul_h1 = [i.unsqueeze(1) for i in [x56, x57, x58, x59, x60, x61]]
        x_mul_v2 = [i.unsqueeze(1) for i in [x10, x11, x12, x13, x14, x15, x16]]
        x_mul_h2 = [i.unsqueeze(1) for i in [x17, x18, x19, x110, x111, x112, x113]]
        x_mul_v3 = [i.unsqueeze(1) for i in [x40, x41]]
        x_mul_h3 = [i.unsqueeze(1) for i in [x42, x43]]
        x_mul_v4 = [i.unsqueeze(1) for i in [x30, x31]]
        x_mul_h4 = [i.unsqueeze(1) for i in [x32, x33]]

        x_mul_v1 = torch.cat(x_mul_v1, dim=1)
        x_mul_h1 = torch.cat(x_mul_h1, dim=1)
        x_mul_v2 = torch.cat(x_mul_v2, dim=1)
        x_mul_h2 = torch.cat(x_mul_h2, dim=1)
        x_mul_v3 = torch.cat(x_mul_v3, dim=1)
        x_mul_h3 = torch.cat(x_mul_h3, dim=1)
        x_mul_v4 = torch.cat(x_mul_v4, dim=1)
        x_mul_h4 = torch.cat(x_mul_h4, dim=1)

        x_mul_v1 = self.subset(x_mul_v1)
        x_mul_h1 = self.subset(x_mul_h1)
        x_mul_v2 = self.subset(x_mul_v2)
        x_mul_h2 = self.subset(x_mul_h2)
        x_mul_v3 = self.subset(x_mul_v3)
        x_mul_h3 = self.subset(x_mul_h3)
        x_mul_v4 = self.subset(x_mul_v4)
        x_mul_h4 = self.subset(x_mul_h4)
        x_mul_1 = torch.cat(
            [x00.unsqueeze(1), x_mul_v1, x_mul_h1, x_mul_v2, x_mul_h2, x_mul_v3, x_mul_h3, x_mul_v4, x_mul_h4], dim=1)
        logit2 = self.fc_2(x_mul_1)
        logit_two_loss = logit2.reshape(B * 17,logit2.size(2))
        logit2_H = nn.functional.softmax(logit2 - torch.max(logit2, dim=-1, keepdim=True)[0].detach(), dim=-1)

        x_pp = x1
        x0_layer4 = self.aggregation(x_pp)
        logit_2 = self.fc_2(x0_layer4)
        x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61 = self.aggregation(
            x_pp[:, :, 0:2, 0:4]), self.aggregation(
            x_pp[:, :, 2:5, 0:4]), self.aggregation(x_pp[:, :, 5:, 0:4]), \
            self.aggregation(x_pp[:, :, 0:2, 4:]), self.aggregation(
            x_pp[:, :, 2:5, 4:]), self.aggregation(x_pp[:, :, 5:, 4:]), self.aggregation(x_pp[:, :, 0:4, 0:2]), \
            self.aggregation(x_pp[:, :, 4:, 0:2]), self.aggregation(x_pp[:, :, 0:4, 2:5]), self.aggregation(
            x_pp[:, :, 4:, 2:5]), \
            self.aggregation(x_pp[:, :, 0:4, 5:]), self.aggregation(x_pp[:, :, 4:, 5:])

        # 1x2
        x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x110, x111, x112, x113 = self.aggregation(
            x_pp[:, :, 0:5, 0:4]), self.aggregation(
            x_pp[:, :, 2:, 0:4]), self.aggregation(x_pp[:, :, 0:5, 4:]), \
            self.aggregation(x_pp[:, :, 2:, 4:]), self.aggregation(
            x_pp[:, :, 0:2, :]), self.aggregation(x_pp[:, :, 2:5, :]), \
            self.aggregation(x_pp[:, :, 5:, :]), self.aggregation(
            x_pp[:, :, :, 0:2]), self.aggregation(x_pp[:, :, :, 2:5]), \
            self.aggregation(x_pp[:, :, :, 5:]), self.aggregation(
            x_pp[:, :, 0:4, 0:5]), self.aggregation(x_pp[:, :, 0:4, 2:]), self.aggregation(
            x_pp[:, :, 4:, 0:5]), self.aggregation(x_pp[:, :, 4:, 2:])
        # 2x2
        x30, x31, x32, x33 = self.aggregation(x_pp[:, :, 0:5, :]), self.aggregation(x_pp[:, :, 2:, :]), \
            self.aggregation(x_pp[:, :, :, 0:5]), self.aggregation(x_pp[:, :, :, 2:])
        # 1x3
        x40, x41, x42, x43 = self.aggregation(x_pp[:, :, :, 0:4]), self.aggregation(x_pp[:, :, :, 4:]), \
            self.aggregation(x_pp[:, :, 0:4, :]), self.aggregation(x_pp[:, :, 4:, :])

        x_mul_v1 = [i.unsqueeze(1) for i in [x50, x51, x52, x53, x54, x55]]
        x_mul_h1 = [i.unsqueeze(1) for i in [x56, x57, x58, x59, x60, x61]]
        x_mul_v2 = [i.unsqueeze(1) for i in [x10, x11, x12, x13, x14, x15, x16]]
        x_mul_h2 = [i.unsqueeze(1) for i in [x17, x18, x19, x110, x111, x112, x113]]
        x_mul_v3 = [i.unsqueeze(1) for i in [x40, x41]]
        x_mul_h3 = [i.unsqueeze(1) for i in [x42, x43]]
        x_mul_v4 = [i.unsqueeze(1) for i in [x30, x31]]
        x_mul_h4 = [i.unsqueeze(1) for i in [x32, x33]]

        x_mul_v1 = torch.cat(x_mul_v1, dim=1)
        x_mul_h1 = torch.cat(x_mul_h1, dim=1)
        x_mul_v2 = torch.cat(x_mul_v2, dim=1)
        x_mul_h2 = torch.cat(x_mul_h2, dim=1)
        x_mul_v3 = torch.cat(x_mul_v3, dim=1)
        x_mul_h3 = torch.cat(x_mul_h3, dim=1)
        x_mul_v4 = torch.cat(x_mul_v4, dim=1)
        x_mul_h4 = torch.cat(x_mul_h4, dim=1)
        x_mul_v1 = self.subset(x_mul_v1)
        x_mul_h1 = self.subset(x_mul_h1)
        x_mul_v2 = self.subset(x_mul_v2)
        x_mul_h2 = self.subset(x_mul_h2)
        x_mul_v3 = self.subset(x_mul_v3)
        x_mul_h3 = self.subset(x_mul_h3)
        x_mul_v4 = self.subset(x_mul_v4)
        x_mul_h4 = self.subset(x_mul_h4)
        x_mul_2 = torch.cat(
            [x0_layer4.unsqueeze(1), x_mul_v1, x_mul_h1, x_mul_v2, x_mul_h2, x_mul_v3, x_mul_h3, x_mul_v4, x_mul_h4],
            dim=1)
        logit3 = self.fc_3(x_mul_2)
        logit_three_loss = logit3.reshape(B * 17, logit3.size(2))
        logit3_H = nn.functional.softmax(logit3 - torch.max(logit3, dim=-1, keepdim=True)[0].detach(), dim=-1)
        logit_H = torch.cat([logit2_H,logit3_H],dim=1)
        x_cat_tran = torch.cat((x_mul_1, x_mul_2), dim=1)
        x_cat = self.part_norm(x_cat_tran)
        x_hgnn_three = self.hnmp_last(x_cat.view(-1, x_mul_2.size(-1)),
                                      logit_H.view(-1, logit_H.size(-1)))
        x_hgnn_two_logit = self.fc_two(x_hgnn_three.view(x.size(0), 34, -1)[:, 0, :])
        x_hgnn_three_logit = self.fc_three(x_hgnn_three.view(x.size(0), 34, -1)[:,17, :])
        x_hgnn_tran = x_hgnn_three.view(x.size(0),34,-1)

        text_emb = self.text_proj_1(Tc)

        sim_1 = x_hgnn_tran @ text_emb.t() * self.logit_scale
        sim_1 = sim_1.reshape(B * 34, sim_1.size(2))
        text_embbing = text_emb.expand(B, -1, -1)
        out_1 = self.transform(text_embbing,x_hgnn_tran)

        text_logit = out_1.mean(-1)
        hash_logit = x_hgnn_two_logit + x_hgnn_three_logit + text_logit
        x_mask = torch.ones(hash_logit.size()).detach().cuda() * 0.7
        for i in range(x_mask.size()[0]):
            x_mask[i, torch.argmax(hash_logit[i])] = 1
        x_b = x_mask * hash_logit
        q_codes = self.fc_q(x_b)
        text_embbed = self.fc_text(text_emb)

        if labels is not None:
            labels_repeat = labels.unsqueeze(1).expand(-1, 17).reshape(-1)
            labels_repeat_contrastive = labels.unsqueeze(1).expand(-1, 34).reshape(-1)
            loss_ce = smooth_CE(hash_logit, labels.view(-1), 1) + \
                      smooth_CE(x_hgnn_three_logit, labels.view(-1), 1) + \
                      smooth_CE(x_hgnn_two_logit, labels.view(-1), 0.9) + \
                      self.clfloss(logit_two_loss, labels_repeat) + \
                      self.clfloss(logit_three_loss, labels_repeat)+ \
                      F.cross_entropy(text_logit, labels) + \
                      F.cross_entropy(sim_1,labels_repeat_contrastive)



            loss = loss_ce
            return hash_logit, q_codes, loss,text_embbed
        else:
            return hash_logit, q_codes