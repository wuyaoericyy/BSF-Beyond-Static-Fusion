import torch.nn as nn
import torch
import math
# form diffurec import DiffuRec # 假设你上面的文件名为 diffurec.py
import torch.nn.functional as F
import copy
import numpy as np
from step_sample import LossAwareSampler
import torch as th


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num + 1 # padding 0
        
        # ID Embedding 层
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        
        self.diffu = diffu  # 这是你扩展后的多模态 DiffuRec 实例
        
        # Loss definitions
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()

    def diffu_pre(self, item_rep_tuple, tag_tuple, mask_seq):
        """
        前向扩散过程
        item_rep_tuple: (seq_id_emb, seq_text, seq_img)
        tag_tuple: (tag_id_emb, tag_text, tag_img)
        """
        seq_rep_diffu, item_rep_out, weights, t, x_t, dy_pre_x_t = self.diffu(item_rep_tuple, tag_tuple, mask_seq)
        return seq_rep_diffu, item_rep_out, weights, t, x_t, dy_pre_x_t 

    def reverse(self, item_rep_tuple, noise_x_t, mask_seq):
        """
        反向生成过程
        item_rep_tuple: (seq_id_emb, seq_text, seq_img) - 用于提供条件(Condition)
        """
        # 注意：这里我们只传入 item_rep_tuple，因为 DiffuRec 的 modal_fusion 会处理它
        # 先手动进行一次 fusion 得到 latent 向量，因为 reverse_p_sample 内部通常需要 latent 形式的条件
        # 但是为了保持 DiffuRec 的封装性，我们最好调用 DiffuRec 内部的方法。
        # 假设 DiffuRec 的 reverse_p_sample 已经被修改为接收原始 tuple 或者 DiffuRec 内部有处理逻辑。
        # 这里建议修改方案：在 Att_Diffuse_model 里调用 fusion，然后传 latent 给 reverse。
        
        # 方案：调用 DiffuRec 的 fusion 接口 (假设我们在 DiffuRec 里暴露了 modal_fusion)
        id_emb, text, img = item_rep_tuple
        # 获取融合后的历史序列表示 [Batch, Seq, Hidden]
        fused_item_rep = self.diffu.modal_fusion(id_emb, text, img)
        
        reverse_pre = self.diffu.reverse_p_sample(fused_item_rep, noise_x_t, mask_seq)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        """
        rep_diffu: 扩散模型输出的 predicted x_0 (Latent Space)
        labels: 真实物品 ID
        此处使用 ID Embedding Matrix 进行打分，这强制 Latent Space 向 ID Space 对齐。
        """
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        scores_pos = scores.gather(1 , labels)  ## labels: b x 1
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1)-scores_pos)/(scores.shape[1]-1)
      
        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))), torch.tensor(1e8))
        return loss   

    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))

    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return scores
    
    def loss_rmse_dy(self, x_t, dy_pre_x_t):
        return torch.sqrt(self.loss_mse(x_t, dy_pre_x_t))
    
    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))
    
    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1) 
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))

    def forward(self, seq_data, tag_data, train_flag=True): 
        """
        seq_data: tuple (seq_ids, seq_text, seq_img)
        tag_data: tuple (tag_ids, tag_text, tag_img)
        """
        # 1. 解包序列数据
        seq_ids, seq_text, seq_img = seq_data
        
        seq_length = seq_ids.size(1)
        mask_seq = (seq_ids > 0).float()

        # 2. 处理 ID Embedding (Seq)
        # Position embedding 可以在这里加，也可以在 fusion 后加，这里为了保持原逻辑在 ID 上加
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=seq_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(seq_ids)
        # position_embeddings = self.position_embeddings(position_ids)

        seq_id_emb = self.item_embeddings(seq_ids)
        seq_id_emb = self.embed_dropout(seq_id_emb)
        # seq_id_emb = seq_id_emb + position_embeddings
        seq_id_emb = self.LayerNorm(seq_id_emb)
        
        # 构建输入的 Tuple
        item_rep_tuple = (seq_id_emb, seq_text, seq_img)
        
        if train_flag:
            # 3. 处理目标数据 (Tag)
            tag_ids, tag_text, tag_img = tag_data
            
            # Tag ID Embedding
            tag_id_emb = self.item_embeddings(tag_ids.squeeze(-1))
            
            # 构建目标的 Tuple
            item_tag_tuple = (tag_id_emb, tag_text, tag_img)

            # 调用 DiffuRec (注意：diffu_pre 现在接收 tuples)
            rep_diffu, rep_item, weights, t, x_t, dy_pre_x_t = self.diffu_pre(item_rep_tuple, item_tag_tuple, mask_seq)
            
            item_rep_dis = None
            seq_rep_dis = None
        else:
            # 4. 推理阶段
            # 随机噪声初始化，形状为 [Batch, Hidden]
            # 注意：这里的形状应该与 fusion 后的 dimension 一致 (args.hidden_size)
            # 我们取 seq_id_emb 的最后一帧作为 shape 参考，因为 Hidden Size 是一样的
            noise_x_t = th.randn_like(seq_id_emb[:, -1, :])
            
            x_t, dy_pre_x_t = None, None
            
            # 调用 reverse，传入序列历史 Tuple
            rep_diffu = self.reverse(item_rep_tuple, noise_x_t, mask_seq)
            
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None

        # 计算分数 (通常用于评估或作为辅助输出，训练loss在外部循环调用 loss_diffu)
        # 这里 scores 还是基于 ID embedding 的投影
        # scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        scores = None 

        return scores, rep_diffu, weights, t, item_rep_dis, seq_rep_dis, x_t, dy_pre_x_t

# 以下保持不变
def create_model_diffu(args):
    # 这里需要确保 DiffuRec 已经被 import 并且 args 里有 text_dim, img_dim 等参数
    diffu_pre = DiffuRec(args)
    return diffu_pre
