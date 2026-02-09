import torch.nn as nn
import torch as th
import torch
import math
import torch.nn.functional as F
import numpy as np

from step_sample import create_named_schedule_sampler # 确保这个文件在你项目的PYTHONPATH中
from recbole.model.abstract_recommender import SequentialRecommender
from timm.models.layers import trunc_normal_
#from recbole.model.sequential_recommender import SequentialRecommender
#from recbole.model.layers import TransformerEncoder # RecBole自带的LayerNorm可能略有不同，但通常兼容


# --------------- Start: Original DiffuRec and supporting functions ---------------

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),  )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,)
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


# RecBole 已经有 LayerNorm，通常可以直接用 nn.LayerNorm 或 recbole.model.layers.LayerNorm
# 这里为了与你的原始代码保持一致，保留你的LayerNorm定义。
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            # RecBole的mask通常是 (batch_size, seq_len)，需要扩展维度以匹配attention score
            # 假设mask是(batch_size, 1, seq_len, seq_len) 或者 (batch_size, seq_len)
            if mask.dim() == 2: # (batch_size, seq_len) for key_padding_mask
                # Convert to additive mask for query x key (batch_size, num_heads, q_len, k_len)
                mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, corr.shape[-2], -1)
                mask = (1.0 - mask.float()) * (-1e9) # Fill padding positions with a very small number
                corr = corr + mask
            elif mask.dim() == 4: # Already a proper attention mask
                corr = corr.masked_fill(mask == 0, -1e9)
        
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout_layer = nn.Dropout(p=dropout) # Changed to avoid name clash with dropout in SublayerConnection

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout_layer(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, config): # Changed args to config for RecBole compatibility
        super(Transformer_rep, self).__init__()
        self.hidden_size = config.hidden_size # Use RecBole's embedding_size for hidden_size
        self.heads = config.attn_heads # Add attn_heads to config
        self.dropout = config.dropout
        self.n_blocks = config.num_blocks # Add num_blocks to config
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class ModalFusion(nn.Module):
    def __init__(self, config): # Changed args to config
        super(ModalFusion, self).__init__()
        self.hidden_size = config.hidden_size
        self.fusion_type = config.fusion_type 
        
        # RecBole config中需定义 item_text_feature_dim 和 item_image_feature_dim
        self.text_input_dim = config.text_size
        self.img_input_dim = config.image_size
        
        # 模态投影层
        # ID embedding 已经是 hidden_size
        self.text_proj = nn.Linear(self.text_input_dim, self.hidden_size)
        self.img_proj = nn.Linear(self.img_input_dim, self.hidden_size)
        
        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                SiLU() # 使用你定义的SiLU
            )
        elif self.fusion_type == 'mlp':
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.hidden_size * 2, self.hidden_size)
            )
        elif self.fusion_type == 'sum':
            self.fusion_layer = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        self.layer_norm = LayerNorm(self.hidden_size) # 使用你定义的LayerNorm

    def forward(self, id_emb, text_feat, img_feat):
        h_text = self.text_proj(text_feat)
        h_img = self.img_proj(img_feat)
        
        if self.fusion_type == 'sum':
            fused = id_emb + h_text + h_img
        else:
            fused = torch.cat([id_emb, h_text, h_img], dim=-1)
            fused = self.fusion_layer(fused)
            
        return self.layer_norm(fused)


class Diffu_xstart(nn.Module):
    def __init__(self, hidden_size, config): # Changed args to config
        super(Diffu_xstart, self).__init__()
        self.hidden_size = hidden_size
        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        self.fuse_linear = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.att = Transformer_rep(config) # Pass config here
        self.lambda_uncertainty = config.lambda_uncertainty # Add to config
        self.dropout = nn.Dropout(config.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t = x_t + emb_t
        
        lambda_uncertainty = th.normal(mean=th.full(rep_item.shape, self.lambda_uncertainty), std=th.full(rep_item.shape, self.lambda_uncertainty)).to(x_t.device)
        
        rep_diffu = self.att(rep_item + lambda_uncertainty * x_t.unsqueeze(1), mask_seq)
        rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))
        out = rep_diffu[:, -1, :]
        
        return out, rep_diffu

class Diffu_xstart_mm(nn.Module):
    def __init__(self, hidden_size, config):
        super(Diffu_xstart_mm, self).__init__()
        self.hidden_size = hidden_size
        
        # [New] 多模态投影层，假设输入视觉/文本维度是 args.vis_dim, args.txt_dim
        # 如果预处理已经对齐了维度，这里可以是简单的 Linear 或者 Identity
        self.image_projector = nn.Sequential(
            nn.Linear(config.image_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(config.text_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        
        self.att = Transformer_rep(config)
        self.lambda_uncertainty = config.lambda_uncertainty
        self.dropout = nn.Dropout(config.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        # ... (保持不变) ...
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    # [Modified] 增加 item_vis, item_txt 输入
    def forward(self, dy_mix_history, x_t, t, mask_seq):
        """
        :param dy_mix_history: [Batch, Seq, Dim] 
               这是从 dynamic_q_sample/dynamic_p_sample 传过来的、
               已经加权混合好了的历史信息 (Weighted Mixed History)
        :param x_t: 带噪的 Target
        """
        
        # 1. 这里的 fused_rep 直接就是外部传进来的混合历史
        # 这样 Denoiser 就明确知道历史上下文长什么样
        fused_rep = dy_mix_history 
        
        # 2. 时间嵌入
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t = x_t + emb_t 
        
        lambda_uncertainty = th.normal(mean=th.full(fused_rep.shape, self.lambda_uncertainty), std=th.full(fused_rep.shape, self.lambda_uncertainty)).to(fused_rep.device)
        
        # 3. Attention
        # 这里的逻辑是：Transformer 看到的是“完美的加权历史”，
        # 它要利用这个历史，去把混在 x_t 里的噪声预测出来。
        rep_diffu = self.att(fused_rep + lambda_uncertainty * x_t.unsqueeze(1), mask_seq)
        rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))
        
        out = rep_diffu[:, -1, :] 
        
        return out, rep_diffu

class ModalGate(nn.Module):
    def __init__(self, hidden_size):
        super(ModalGate, self).__init__()
        # 输入是 ID 历史的聚合 (dy_x_s_1)，输出是 3 个权重 (ID, Vis, Txt)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 3), # 输出 3 个 logits
            nn.Softmax(dim=-1) # 归一化，保证和为 1
        )

    def forward(self, history_rep):
        return self.net(history_rep)

class SequenceAggregator_0(nn.Module):
    def __init__(self, hidden_size, num_heads=1, num_layers=1):
        super().__init__()
        # 可以是轻量级 Transformer 或简单的平均池化
        # 这里使用一个简单的TransformerEncoderLayer作为示例
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1) # 用于将序列池化为向量

    def forward(self, seq_embeddings, mask_seq=None):
        # seq_embeddings: [B, S, D]
        # mask_seq: [B, S] -> Transformer mask: [B, S] or [B, 1, S] or [B, S, S]
        
        # TransformerEncoderLayer mask is typically (Batch, Head, Seq_len, Seq_len)
        # Or src_key_padding_mask: [B, S] (True if padded)
        
        # Invert mask for Transformer: True for padded positions
        if mask_seq is not None:
             src_key_padding_mask = (mask_seq == 0) # True for padded elements
        else:
             src_key_padding_mask = None

        # Pass through Transformer encoder
        # Transformer requires (batch, seq_len, dim)
        if self.transformer_encoder:
            # For TransformerEncoder, mask is usually True for padding positions
            # And has shape (N, S) where N is batch size, S is sequence length
            # Convert mask_seq ([B,S] 0 for padding) to src_key_padding_mask ([B,S] True for padding)
            agg_output = self.transformer_encoder(seq_embeddings, src_key_padding_mask=src_key_padding_mask)
        else: # Fallback to simple mean pooling if no transformer
            agg_output = seq_embeddings

        # Pool to get sequence-level representation [B, D]
        # Transpose for AdaptiveAvgPool1d input [B, D, S] -> output [B, D, 1]
        pooled_output = self.pool(agg_output.transpose(1, 2)).squeeze(-1) # [B, D]
        return pooled_output

class SequenceAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        # 不需要任何可学习的参数或复杂的层
        pass

    def forward(self, seq_embeddings, mask_seq=None):
        """
        seq_embeddings: [Batch, Seq_Len, Hidden_Size]
        mask_seq: [Batch, Seq_Len] (1 for valid, 0 for padding)
        """
        if mask_seq is None:
            # 如果没有 Mask，直接对 dim=1 (Seq_Len) 求平均
            return seq_embeddings.mean(dim=1) 
        
        # 如果有 Mask，我们需要计算 Masked Mean
        # mask_seq: [B, S] -> [B, S, 1] for broadcasting
        mask = mask_seq.unsqueeze(-1).float() 
        
        # Sum of embeddings for valid tokens
        sum_embeddings = torch.sum(seq_embeddings * mask, dim=1) # [B, D]
        
        # Count of valid tokens
        # Add a small epsilon to avoid division by zero
        valid_counts = torch.sum(mask, dim=1).clamp(min=1e-9) # [B, 1]
        
        # Mean Pooling
        pooled_output = sum_embeddings / valid_counts # [B, D]
        
        return pooled_output




class DiffuRec(nn.Module):
    def __init__(self, config): # Changed args to config
        super(DiffuRec, self).__init__()
        self.hidden_size = config.hidden_size
        self.schedule_sampler_name = config.schedule_sampler_name # Add to config
        self.diffusion_steps = config.diffusion_steps # Add to config
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])

        self.noise_schedule = config.noise_schedule # Add to config
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
        betas = np.array(betas, dtype=np.float64)
        self.eta = config.eta # Add to config
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.delta = config.delta # Add to config
        self.dy_t = np.array([1-self.delta + 2*self.delta/(config.diffusion_steps-1) * i for i in range(config.diffusion_steps)])
        
        self.gammas_t = self.eta * self.alphas_cumprod + (1.0 - self.eta)
        self.sqrt_gammas_t = np.sqrt(self.gammas_t)
        self.sqrt_one_minus_gammas_t = np.sqrt(1.0 - self.gammas_t)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
       
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = config.rescale_timesteps # Add to config
        self.original_num_steps = len(betas)

        # [NEW] 初始化多模态融合层
        self.fusion_type = config.fusion_type
        if self.fusion_type != 'dydiff':
            self.modal_fusion = ModalFusion(config) # Pass config
            self.xstart_model = Diffu_xstart(self.hidden_size, config) # Pass config
        else:
            #self.text_input_dim = config.text_size
            #self.img_input_dim = config.image_size
            #self.text_proj = nn.Linear(self.text_input_dim, self.hidden_size)
            #self.img_proj = nn.Linear(self.img_input_dim, self.hidden_size)
            self.modal_gate = ModalGate(self.hidden_size)
            self.xstart_model = Diffu_xstart_mm(self.hidden_size, config) # Pass config
        
        self.mc_loss = config.mc_loss

        # [New] Initialize Sequence Aggregators for Rep_v and Rep_t
        self.vis_seq_aggregator = SequenceAggregator()
        self.txt_seq_aggregator = SequenceAggregator()
    
        # [New] Temperature for InfoNCE Loss
        self.temperature = config.temperature 
            

        


    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        return betas
    

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
        if mask is not None:
            # Mask should be 1 where active, 0 where padded
            # We want to keep x_start at masked (0) positions, and x_t at unmasked (1) positions
            # Assuming mask_seq is (Batch, Seq_Len)
            # Expand mask_seq to match x_t's shape for element-wise operation
            expanded_mask = mask.unsqueeze(-1).expand_as(x_t)
            # Use torch.where to keep original x_start if mask is 0 (padded position),
            # otherwise use the noisy x_t
            # This is tricky because `mask` for q_sample means "anchoring masked position"
            # It's usually a 1 where original value should be preserved and 0 where it should be diffused.
            # Your original code used mask==0 to preserve x_start, so let's stick to that.
            x_t = torch.where(expanded_mask==0, x_start, x_t)
        return x_t
    
    def dynamic_q_sample_1(self, item_list, x_start, t, noise=None, mask=None):
        if noise is None:
            target_noise = th.randn_like(x_start)
            item_list_noise = th.randn_like(item_list)

        dy_t = _extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1)))
        cumprod_dy_t = dy_t * torch.cumprod(_extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1))), dim=1)
        flip_log_dy_t = torch.flip(torch.log(cumprod_dy_t + 1), dims=[1]) * mask 
        #max = torch.max(flip_log_dy_t,dim=1).values.unsqueeze(-1)
        #norm_flip_log_dy_t = flip_log_dy_t / max
        dy_x_s_1 = torch.sum(flip_log_dy_t.unsqueeze(-1) * item_list, dim=1)
        dy_noise_s_1 = torch.sum(flip_log_dy_t.unsqueeze(-1) * item_list_noise, dim=1)
        dy_x_0 = dy_x_s_1 + _extract_into_tensor(self.dy_t, t, x_start.shape) * x_start
        dy_noise = dy_noise_s_1 + _extract_into_tensor(self.dy_t, t, target_noise.shape) * target_noise 

        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, dy_x_0.shape) * dy_x_0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, dy_noise.shape)
            * dy_noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick


        return x_t, dy_x_s_1, dy_noise_s_1

        # Ensure mask is appropriate shape for item_list (Batch, Seq_Len)
        # Assuming mask here refers to the sequence padding mask (1 for active, 0 for padding)
        # So we only consider active items in item_list
        # If mask is None, then treat all sequence items as active (full sequence length)
        if mask is None:
            mask = torch.ones(item_list.shape[0], item_list.shape[1], device=item_list.device)
        
        dy_t_tensor = _extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1)))
        
        # Ensure cumprod works correctly for each sequence independently
        # and respect padding.
        # We need a custom cumprod that handles masks, or apply it before masking.
        # Let's adjust cumprod_dy_t to be per-sequence
        
        # Calculate `cumprod_dy_t` for each sequence, then apply the mask.
        cumprod_dy_t_raw = torch.cumprod(dy_t_tensor, dim=1)
        # If the first element is dy_t[t], then cumprod[0] is dy_t[t].
        # If it's a sequence-wise cumulative product, then dy_t_tensor[i, j] is dy_t for time t at sequence position j.
        # The logic `dy_t * torch.cumprod(...)` seems to imply a global time step `t` is applied to each element.
        # Let's assume `_extract_into_tensor(self.dy_t, t, (B, S))` means `dy_t` value for time `t` replicated `S` times per batch.
        # This makes `dy_t_tensor` a (Batch, Seq_len) tensor where each row is `[dy_t[t], dy_t[t], ..., dy_t[t]]`.
        # This seems incorrect for a sequence of varying `dy_t` weights for `s_1` terms.
        # The original `dy_t` is indexed by `i` (0 to diffusion_steps-1).
        # This `dy_t` is intended to weight the sequence items based on their position/age, not the diffusion timestep `t`.
        # Your current `dy_t` extraction means `dy_t[t]` is a scalar, broadcasted.
        # If dy_t is meant to scale influence of historical items in a sequence,
        # it should probably be indexed by sequence position `j` or some relative age.
        # For multi-modal, this dynamic part should still be based on the fused representations.
        
        # Let's assume the current logic for `_extract_into_tensor(self.dy_t, t, ...)`
        # is intended to apply a global weight `dy_t[t]` to each item in the sequence.
        # If it was meant for sequence position, the first argument would be a (Batch, Seq_len) tensor of indices.

        # Original: `cumprod_dy_t = dy_t * torch.cumprod(_extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1))), dim=1)`
        # This looks like `dy_t[t] * (dy_t[t])^j` which is odd.
        # Let's stick to your previous code's intent as closely as possible.
        # Assuming `_extract_into_tensor(self.dy_t, t, ...)` means `self.dy_t[t]` value.
        # This implies `dy_t_tensor` is a tensor of shape `(B, S)` where all values are `self.dy_t[t]`.
        # The `cumprod_dy_t` then becomes `dy_t[t] * (dy_t[t])^j`. This looks strange if it's meant to be independent.
        # A more common interpretation for `dy_t` as sequence weights is:
        # `dy_t_weights = _extract_into_tensor(self.dy_t, sequence_indices_for_weights, (B,S))`
        # For now, I will use your exact original logic for `dy_x_s_1` and `dy_noise_s_1` assuming `_extract_into_tensor(self.dy_t, t, (B, S))` is a `(B, S)` tensor
        # where each element is `self.dy_t[t]`. This effectively means `dy_t[t]` is broadcasted.

        # Let's adjust `cumprod_dy_t` and `flip_log_dy_t` based on sequence elements.
        # If `dy_t` is constant for a given `t`, then `cumprod` over a constant value is `const^idx`.
        # Original: dy_t = _extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1)))
        # This `dy_t` is essentially `self.dy_t[t]` broadcasted. Let's name it `dy_t_val_t`.
        dy_t_val_t = _extract_into_tensor(self.dy_t, t, (item_list.size(0), 1)) # (Batch, 1)

        # Create a tensor for cumulative product that handles padding
        # For a sequence [a,b,c,0,0], if dy_t_val_t is `w`, we want to compute terms like:
        # w*a, w*w*b, w*w*w*c
        # We can construct weights for each position then apply element-wise.
        seq_len = item_list.shape[1]
        indices = torch.arange(seq_len, device=item_list.device, dtype=torch.float32).unsqueeze(0) # (1, seq_len)
        # cumprod_dy_t_weights will be (Batch, Seq_len) where element (i,j) is (dy_t_val_t[i])^(j+1)
        cumprod_dy_t_weights = dy_t_val_t.pow(indices + 1) # (Batch, 1) ^ (1, Seq_len) -> (Batch, Seq_len)

        # Apply mask and then sum
        # `torch.flip(..., dims=[1])` is used to weight recent items more.
        # Let's assume `mask` is 1 for valid items, 0 for padded ones.
        # The sum should only include valid (non-padded) elements.
        
        # `flip_log_dy_t = torch.flip(torch.log(cumprod_dy_t + 1), dims=[1]) * mask`
        # This `cumprod_dy_t` was `dy_t_val_t * cumprod_dy_t_raw`.
        # This original formulation is slightly ambiguous.
        # A more straightforward interpretation for dynamic_q_sample_1 as sequence-dependent is often a weighted sum.
        # Assuming `dy_t` is a scaling factor for `x_0` in the current diffusion step.
        # And `dy_x_s_1` is a weighted sum of historical items.

        # Let's refine based on the common interpretation for dynamic q-sampling in seq rec:
        # weighted_item_list = item_list * dynamic_weights.unsqueeze(-1)
        # dy_x_s_1 = weighted_item_list.sum(dim=1) / sum(dynamic_weights)

        # For strict adherence to your code's arithmetic:
        # `dy_t` is `_extract_into_tensor(self.dy_t, t, ...)` which is `self.dy_t[t]`
        # So `dy_t` is a scalar (broadcasted).
        # `cumprod_dy_t` in your original code becomes `self.dy_t[t] * (self.dy_t[t])^j` or similar.
        # This suggests `self.dy_t` array defines how each previous item contributes.
        # The `t` here refers to the diffusion step `t`, not sequence position.
        # So `dy_t = self.dy_t[t]`.

        # Let's assume the simplest interpretation where dy_t is a scalar from the schedule.
        # `cumprod_dy_t` and `flip_log_dy_t` are difficult to interpret this way.
        # It seems `_extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1)))` is intended to give a different weight for each item in the sequence.
        # If `t` is actually `sequence_length - current_position_in_seq` for this part, that would make sense.
        # Given `t` is a single scalar per batch element (diffusion time), the `_extract_into_tensor` makes `dy_t` a scalar repeated.

        # To align with multi-modal and make it work with RecBole, we'll keep `item_list` and `x_start` as fused embeddings.
        # The `t` is the diffusion timestep.
        # `dy_t` is `self.dy_t[t]`.

        # Let's reformulate `dy_x_s_1` and `dy_noise_s_1` to avoid the ambiguous cumprod logic if `dy_t` is just `self.dy_t[t]`.
        # It's usually a simple sum of previous items, weighted by some function of their position or age in the sequence.
        
        # If `dy_t` is intended to be a sequence-position dependent weight, `t` needs to be `(Batch, Seq_Len)`.
        # But `t` is `(Batch,)`. So `dy_t` is broadcasted.
        # This implies `dy_t_val_at_t = _extract_into_tensor(self.dy_t, t, item_list.shape).squeeze(-1)` if you just want to multiply.
        
        # Sticking to the most literal translation of your exact code arithmetic logic for now:
        dy_t_for_sequence = _extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1))) # (Batch, Seq_Len) all equal values `self.dy_t[t]`

        cumprod_dy_t = torch.ones_like(dy_t_for_sequence) # Initialize with ones
        # For each sequence, perform cumulative product and apply mask
        for i in range(item_list.shape[0]):
            valid_len = int(mask[i].sum().item())
            if valid_len > 0:
                # Apply cumprod only to valid part of sequence
                # Assuming dynamic_weights decrease influence of older items: flip before cumprod
                flipped_valid_dy_t = torch.flip(dy_t_for_sequence[i, :valid_len], dims=[0])
                cumprod_on_flipped = torch.cumprod(flipped_valid_dy_t, dim=0)
                # Flip back to restore original order
                cumprod_dy_t[i, :valid_len] = torch.flip(cumprod_on_flipped, dims=[0])
            # For padded parts, cumprod_dy_t remains 1 or 0 based on initialization.
        
        # `flip_log_dy_t = torch.flip(torch.log(cumprod_dy_t + 1), dims=[1]) * mask`
        # This uses the cumprod result directly.
        flip_log_dy_t = torch.flip(torch.log(cumprod_dy_t + 1), dims=[1]) * mask.float() # Ensure mask is float

        # Sum up for dy_x_s_1 and dy_noise_s_1
        dy_x_s_1 = torch.sum(flip_log_dy_t.unsqueeze(-1) * item_list, dim=1)
        dy_noise_s_1 = torch.sum(flip_log_dy_t.unsqueeze(-1) * item_list_noise, dim=1)
        
        # The rest of the dynamic q_sample_1 logic
        dy_x_0 = dy_x_s_1 + _extract_into_tensor(self.dy_t, t, x_start.shape) * x_start
        dy_noise = dy_noise_s_1 + _extract_into_tensor(self.dy_t, t, target_noise.shape) * target_noise 

        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, dy_x_0.shape) * dy_x_0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, dy_noise.shape)
            * dy_noise
        )
        return x_t, dy_x_s_1, dy_noise_s_1
    

    def dynamic_q_sample_2(self, item_tuple, target_tuple, t, noise=None, mask=None):
        item_list = item_tuple[0]
        text_list = item_tuple[1]
        image_list = item_tuple[2]
        x_start = target_tuple[0]
        if noise is None:
            target_noise = th.randn_like(x_start)
            item_list_noise = th.randn_like(item_list)
        

        dy_t = _extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1)))
        cumprod_dy_t = dy_t * torch.cumprod(_extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1))), dim=1)
        flip_log_dy_t = torch.flip(torch.log(cumprod_dy_t + 1), dims=[1]) * mask 
        #max = torch.max(flip_log_dy_t,dim=1).values.unsqueeze(-1)
        #norm_flip_log_dy_t = flip_log_dy_t / max

        # 0. 投影视觉和文本
        proj_text = self.xstart_model.text_projector(text_list)
        proj_img = self.xstart_model.image_projector(image_list)

        # 1. 计算各模态的基础历史聚合
        dy_x_s_1 = torch.sum(flip_log_dy_t.unsqueeze(-1) * item_list, dim=1)
        dy_text_s_1 = torch.sum(flip_log_dy_t.unsqueeze(-1) * proj_text, dim=1)
        dy_img_s_1 = torch.sum(flip_log_dy_t.unsqueeze(-1) * proj_img, dim=1)

        # 2. [New] 动态生成权重
        # 利用 ID 的历史聚合信息 dy_x_s_1 来判断当前应该关注哪个模态
        # 例如：如果历史 ID 聚合出来的信息很混乱（熵很高），Gate 可能会给 Vis 更高权重
        modal_weights = self.modal_gate(dy_x_s_1)
        w_id = modal_weights[:, 0].view(-1, 1, 1) # [B, 1, 1] 广播到序列
        w_text = modal_weights[:, 1].view(-1, 1, 1)
        w_img = modal_weights[:, 2].view(-1, 1, 1)

         # 3. 【核心修正】：构造传给 Transformer 的 Mixed Sequence [B, S, D]
        # 我们不是把 sum 后的结果传给 Transformer，而是把加权后的序列传给它
        mixed_seq_rep = w_id * item_list + w_text * proj_text + w_img * proj_img
        
        # 4. 加权融合
        # dy_mix_history = w_id * dy_x_s_1 + w_vis * dy_v_s_1 + w_txt * dy_t_s_1
        # 通常我们希望保持 ID 为主导，模态为辅助，所以也可以写成残差形式：
        # dy_mix_history = dy_x_s_1 + w_vis * dy_v_s_1 + w_txt * dy_t_s_1 (看实验效果)
        
        # 这里使用全动态加权：
        dy_mix_history = w_id.squeeze(1) * dy_x_s_1 + w_img.squeeze(1) * dy_img_s_1 + w_text.squeeze(1) * dy_text_s_1
        
        dy_x_0 = dy_mix_history + _extract_into_tensor(self.dy_t, t, x_start.shape) * x_start
        

        # 噪声部分保持原逻辑 (通常我们只对 ID 空间加噪，模态作为 Condition 视为确定的 Trigger)
        dy_noise_s_1 = torch.sum(flip_log_dy_t.unsqueeze(-1) * item_list_noise, dim=1)
        dy_noise = dy_noise_s_1 + _extract_into_tensor(self.dy_t, t, target_noise.shape) * target_noise 

        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, dy_x_0.shape) * dy_x_0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, dy_noise.shape)
            * dy_noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick


        return x_t, mixed_seq_rep, dy_mix_history, dy_noise_s_1


    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq):
        model_output, _ = self.xstart_model(rep_item, x_t, self._scale_timesteps(t), mask_seq)
        
        x_0 = model_output
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance
    
    def dynamic_p_mean_variance_1(self, rep_item, x_t, t, mask_seq): # Removed dy_x_s_1 as direct input
        # Recalculate dy_x_s_1 here as it's needed for x_0 calculation if `x_0 = (model_output - dy_x_s_1) / dy_t`
        # Or remove it if `model_output` is directly `x_0`.
        # Your current logic for x_0 = dy_pre_x_t (model_output) means dy_x_s_1 is NOT used in the final x_0 calculation.
        # But if you want to use it, you'd need to recompute `dy_x_s_1` here.

        # Let's assume model_output is directly the predicted x_0 for simplicity as in your forward.
        model_output, _ = self.xstart_model(rep_item, x_t, self._scale_timesteps(t), mask_seq)
        
        x_0 = model_output # Your forward uses this line `x_0 = dy_pre_x_t`
        
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance
    
    # [Modified DiffuRec.dynamic_p_mean_variance] 接收 mixed_seq_rep 和 dy_mix_history_agg
    def dynamic_p_mean_variance_mm(self, mixed_seq_rep, x_t, t, mask_seq, dy_mix_history_agg):
        """
        :param mixed_seq_rep: [Batch, Seq_Len, Hidden_Size] - 混合了ID, Vis, Txt的序列表示
        :param x_t: [Batch, Hidden_Size] - 当前时间步的带噪目标向量
        :param t: [Batch] - 时间步
        :param mask_seq: [Batch, Seq_Len] - 序列掩码
        :param dy_mix_history_agg: [Batch, Hidden_Size] - 用于后验均值计算的聚合量
        """
        # 模型预测 (Denoiser) 期望 mixed_seq_rep 作为输入序列
        model_output, _ = self.xstart_model(mixed_seq_rep, x_t, self._scale_timesteps(t), mask_seq)
        
        # model_output 是预测的 x_0 (一个聚合向量)
        x_0_predicted = model_output
        
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        # 这里使用预测的 x_0_predicted 作为 q(x_{t-1}|x_t, x_0) 中的 x_0
        # 假设 dy_mix_history_agg 已经融入到 x_0_predicted 的学习中
        model_mean = self.q_posterior_mean_variance(x_start=x_0_predicted, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq)
        noise = th.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise
        return sample_xt
    
    def dynamic_p_sample_1(self, item_rep, noise_x_t, t, mask_seq):
        # The `dy_x_s_1` calculation here is specific to your model
        # Replicated from dynamic_q_sample_1 for consistency
        dy_t_for_sequence = _extract_into_tensor(self.dy_t, t, (item_rep.size(0), item_rep.size(1)))
        
        cumprod_dy_t = torch.ones_like(dy_t_for_sequence)
        for i in range(item_rep.shape[0]):
            valid_len = int(mask_seq[i].sum().item())
            if valid_len > 0:
                flipped_valid_dy_t = torch.flip(dy_t_for_sequence[i, :valid_len], dims=[0])
                cumprod_on_flipped = torch.cumprod(flipped_valid_dy_t, dim=0)
                cumprod_dy_t[i, :valid_len] = torch.flip(cumprod_on_flipped, dims=[0])
        
        flip_log_dy_t = torch.flip(torch.log(cumprod_dy_t + 1), dims=[1]) * mask_seq.float()

        # dy_x_s_1 is calculated but *not* directly used by dynamic_p_mean_variance_1 if x_0 = model_output
        dy_x_s_1 = torch.sum(flip_log_dy_t.unsqueeze(-1) * item_rep, dim=1) # Needed if x_0 is computed like `(model_output - dy_x_s_1) / dy_t`

        model_mean, model_log_variance = self.dynamic_p_mean_variance_1(item_rep, noise_x_t, t, mask_seq)
        noise = th.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise
        return sample_xt
    
    def dynamic_p_sample_mm(self, item_rep_tuple, noise_x_t, t, mask_seq):
        """
        Performs one reverse diffusion step (sampling x_{t-1} from x_t).

        :param item_list: [B, S, D_id] - Historical ID embedding sequence.
        :param item_vis: [B, S, D_input_modality] - Historical visual feature sequence.
        :param item_txt: [B, S, D_input_modality] - Historical text feature sequence.
        :param noise_x_t: [B, D] - Noisy target latent at time t.
        :param t: [Batch] - Current diffusion timestep.
        :param mask_seq: [B, S] - Sequence mask.
        :return: sample_xt: [B, D] - Sampled target latent at time t-1.
        """
        item_list = item_rep_tuple[0]
        item_txt = item_rep_tuple[1]
        item_vis = item_rep_tuple[2]

        # 1. 投影视觉和文本特征到相同的 hidden_size 维度 [B, S, D]
        proj_vis = self.xstart_model.image_projector(item_vis) 
        proj_txt = self.xstart_model.text_projector(item_txt)

        # 2. 计算时间相关的权重 (用于 Gate 的输入和聚合)
        dy_t = _extract_into_tensor(self.dy_t, t, (item_list.size(0), item_list.size(1)))
        cumprod_dy_t = dy_t * torch.cumprod(dy_t, dim=1)
        flip_log_dy_t = torch.flip(torch.log(cumprod_dy_t + 1), dims=[1]) * mask_seq

        # 3. 计算用于 ModalGate 的历史聚合向量 [B, D]
        dy_x_s_1_agg_for_gate = torch.sum(flip_log_dy_t.unsqueeze(-1) * item_list, dim=1)

        # 4. 动态生成模态权重 [B, 3]
        weights = self.modal_gate(dy_x_s_1_agg_for_gate)
        w_id = weights[:, 0].view(-1, 1, 1) 
        w_vis = weights[:, 1].view(-1, 1, 1)
        w_txt = weights[:, 2].view(-1, 1, 1)

        # 5. 构造传给 Denoiser (Transformer) 的 Mixed Sequence [B, S, D]
        # 这个序列包含了当前步的动态模态混合信息
        mixed_seq_rep = w_id * item_list + w_vis * proj_vis + w_txt * proj_txt
        
        # 6. 计算用于 Diffusion 后验均值公式的聚合量 [B, D]
        # 这是为了计算 q(x_{t-1} | x_t, x_0) 中的 x_0_predicted 所需的聚合背景
        dy_x_s_1_agg_for_diffusion = torch.sum(flip_log_dy_t.unsqueeze(-1) * item_list, dim=1)
        dy_v_s_1_agg_for_diffusion = torch.sum(flip_log_dy_t.unsqueeze(-1) * proj_vis, dim=1)
        dy_t_s_1_agg_for_diffusion = torch.sum(flip_log_dy_t.unsqueeze(-1) * proj_txt, dim=1)
        
        dy_mix_history_agg = w_id.squeeze(1) * dy_x_s_1_agg_for_diffusion + \
                             w_vis.squeeze(1) * dy_v_s_1_agg_for_diffusion + \
                             w_txt.squeeze(1) * dy_t_s_1_agg_for_diffusion

        # 7. 调用 dynamic_p_mean_variance 计算模型预测的均值和方差
        # 传入 mixed_seq_rep 作为 Transformer 的输入序列，以及 dy_mix_history_agg 作为聚合量
        model_mean, model_log_variance = self.dynamic_p_mean_variance_mm(
            mixed_seq_rep, noise_x_t, t, mask_seq, dy_mix_history_agg
        )
        
        # 8. 从均值和方差中采样得到 x_{t-1}
        noise = th.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1)))) # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise
        return sample_xt


    def reverse_p_sample(self, fused_item_rep, noise_x_t, mask_seq):
        device = noise_x_t.device # Use the device of the input noise
        indices = list(range(self.num_timesteps))[::-1]
        
        # Fused_item_rep is already the multi-modal sequence representation
        for i in indices:
            t = th.tensor([i] * fused_item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.dynamic_p_sample_1(fused_item_rep, noise_x_t, t, mask_seq)
        return noise_x_t
    
    def reverse_p_sample_mm(self, item_rep_tuple, noise_x_t, mask_seq):
        device = noise_x_t.device # Use the device of the input noise
        indices = list(range(self.num_timesteps))[::-1]
        
        # Fused_item_rep is already the multi-modal sequence representation
        for i in indices:
            t = th.tensor([i] * item_rep_tuple[0].shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.dynamic_p_sample_mm(item_rep_tuple, noise_x_t, t, mask_seq)
        return noise_x_t 

    def forward(self, item_rep_tuple, item_tag_tuple, mask_seq):        
        # item_rep_tuple: (id_seq_emb, text_seq_feat, img_seq_feat)
        # item_tag_tuple: (id_tag_emb, text_tag_feat, img_tag_feat)
        
        

        # 1. 执行模态融合
        if self.fusion_type != 'dydiff':
            item_rep = self.modal_fusion(*item_rep_tuple) # (Batch, Seq_Len, Hidden)
            item_tag = self.modal_fusion(*item_tag_tuple) # (Batch, Hidden)
            t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_rep.device) # use item_rep.device
        else:
            t, weights = self.schedule_sampler.sample(item_rep_tuple[0].shape[0], item_rep_tuple[0].device)
        
        #noise = th.randn_like(item_tag)
        

        if self.fusion_type != 'dydiff':
            x_t, dy_x_s_1, dy_noise_s_1 = self.dynamic_q_sample_1(item_rep, item_tag, t, mask=mask_seq)
            dy_pre_x_t, dy_item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)
        else:
            x_t, mixed_seq_rep, dy_mix_history, dy_noise_s_1 = self.dynamic_q_sample_2(item_rep_tuple, item_tag_tuple, t, mask=mask_seq)
            dy_pre_x_t, dy_item_rep_out = self.xstart_model(mixed_seq_rep, x_t, self._scale_timesteps(t), mask_seq)

        
        
        

        x_0 = dy_pre_x_t # The predicted x_0 is directly model_output
        item_rep_out = dy_item_rep_out # This is the full sequence latent representation from Transformer_rep in xstart_model

        # Prepare features for L_MC calculation
        # Project actual multimodal features for the target item (S)
        # Note: item_vis and item_txt are sequences, we need the S-th item
        item_vis = item_rep_tuple[2]
        item_txt = item_rep_tuple[1]
        e_v_S = self.xstart_model.image_projector(item_vis[:, -1, :]) # Assuming item_vis is [B, S, D_raw], takes last item
        e_t_S = self.xstart_model.text_projector(item_txt[:, -1, :]) # [B, D]

        # Aggregate historical sequences (0 to S-1)
        # Note: item_vis and item_txt passed here are full sequences [B, S, D_raw]
        # And their projected versions will be handled by SequenceAggregator if it has projectors
        # Or project first then aggregate
        proj_vis_seq = self.xstart_model.image_projector(item_vis) # [B, S, D]
        proj_txt_seq = self.xstart_model.text_projector(item_txt) # [B, S, D]
        
        Rep_v = self.vis_seq_aggregator(proj_vis_seq[:, :-1, :], mask_seq[:, :-1]) # Aggregate history up to S-1
        Rep_t = self.txt_seq_aggregator(proj_txt_seq[:, :-1, :], mask_seq[:, :-1]) # [B, D]

        #return x_0, item_rep_out, weights_sampler, t, x_t, dy_pre_x_t, modal_weights, e_v_S, e_t_S, Rep_v, Rep_t

        return x_0, item_rep_out, weights, t, x_t, dy_pre_x_t, e_v_S, e_t_S, Rep_v, Rep_t




# --------------- End: Original DiffuRec and supporting functions ---------------

class MM_TD_Rec_MC(SequentialRecommender):
    """
    MultiModalDiffusionRec is a sequential recommender that utilizes a Diffusion Model
    with multi-modal (ID, Text, Image) item representations for next item prediction.
    It integrates the DiffuRec and ModalFusion components.
    """

    def __init__(self, config, dataset, index_to_id, index_to_keys, num_patches, text_emb=None, image_emb=None):
        super(MM_TD_Rec_MC, self).__init__(config, dataset)

        # 1. RecBole config parameters
        self.config = config
        self.hidden_size = config.hidden_size
        self.loss_type = config["loss_type"]
        self.item_num = self.n_items # Total number of items including padding
        self.device = config["device"]

        # Dropouts
        self.dropout_prob = config["dropout_prob"]
        self.emb_dropout = config.emb_dropout
        self.dropout = config.dropout

        # Diffusion parameters (from config)
        self.diffusion_steps = config.diffusion_steps
        self.noise_schedule = config.noise_schedule
        self.schedule_sampler_name = config.schedule_sampler_name
        self.eta = config.eta
        self.delta = config.delta
        self.rescale_timesteps = config.rescale_timesteps
        self.lambda_uncertainty = config.lambda_uncertainty
        self.attn_heads = config.attn_heads
        self.num_blocks = config.num_blocks
        self.fusion_type = config.fusion_type

        # 2. RecBole Item Embeddings and Features
        self.id_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        

        self.image_size = image_emb.shape[-1]
        self.text_size = text_emb.shape[-1]
        config.image_size = self.image_size
        config.text_size = self.text_size
        self.text_emb = text_emb.to(self.config['device'])
        self.image_emb = image_emb.to(self.config['device'])
        if self.config['contrasive_loss']:
            self.temp = config["temperature"]
        self.mc_loss = config.mc_loss
        self.temperature = config["temperature"]
        self.lambda3 = config.lambda3

        if self.config['if_dual_pos_embed']:
            self.id_dual_pos_embed = nn.Parameter(torch.zeros(1, config['MAX_ITEM_LIST_LENGTH'], self.hidden_size)).to(self.config['device'])
            trunc_normal_(self.id_dual_pos_embed, std=.02)
            self.image_dual_pos_embed = nn.Parameter(torch.zeros(1, config['MAX_ITEM_LIST_LENGTH'], self.image_size)).to(self.config['device'])
            trunc_normal_(self.image_dual_pos_embed, std=.02)
            self.text_dual_pos_embed = nn.Parameter(torch.zeros(1, config['MAX_ITEM_LIST_LENGTH'], self.text_size)).to(self.config['device'])
            trunc_normal_(self.text_dual_pos_embed, std=.02)

        # Check if text/image features are actually provided if specified by fusion type
        if self.fusion_type != 'sum' or self.text_emb is None or self.image_emb is None:
            if self.text_emb is None and self.image_emb is None:
                # If no multimodal features, treat text/image as zero vectors or identical to ID for 'sum'
                # For `ModalFusion` to work, text/img features should not be None.
                # If only ID, then set text/img proj to identity and feed zero/ID.
                # Or simplify ModalFusion for ID-only case.
                pass # The ModalFusion constructor will handle text_input_dim/img_input_dim.
                     # We will feed zero tensors if features are not available.


        # 3. Initialize DiffuRec (your core diffusion model)
        self.diffu = DiffuRec(config) # Pass the full config object

        # 4. Loss functions
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # LayerNorm after sequence embeddings (ID + pos)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12) # Use your LayerNorm
        self.dropout_layer = nn.Dropout(config.dropout)


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, next_item):
        # 2. Get multi-modal features for sequence and target
        # Sequence features
        seq_id_emb = self.id_embedding(item_seq)
        #id_emb = self.dropout(id_emb)
        #id_emb = self.id_LayerNorm(id_emb)
        

        seq_text_feat = self.text_emb[item_seq]
        seq_img_feat = self.image_emb[item_seq]
        tag_id_emb = self.id_embedding(next_item)
        tag_text_feat = self.text_emb[next_item]
        tag_img_feat = self.image_emb[next_item]
        
        # Add position embeddings
        if self.config['if_dual_pos_embed']:
            seq_id_emb = seq_id_emb + self.id_dual_pos_embed
            seq_text_feat = seq_text_feat + self.text_dual_pos_embed
            seq_img_feat = seq_img_feat + self.image_dual_pos_embed
            #position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=self.device).unsqueeze(0)
            #position_embeddings = self.position_embedding(position_ids)
            #seq_id_emb = seq_id_emb + position_embeddings # Add position to ID embedding before fusion
        seq_id_emb = self.LayerNorm(seq_id_emb)
        seq_id_emb = self.dropout_layer(seq_id_emb) # Apply dropout after layer norm
        
        item_rep_tuple = (seq_id_emb, seq_text_feat, seq_img_feat) # (Batch, Seq_Len, Dim)

        # Target item features (next_item)
        item_tag_tuple = (tag_id_emb, tag_text_feat, tag_img_feat) # (Batch, Dim)


        # 3. Prepare sequence mask
        # RecBole's compute_mask can create padding mask for sequences
        # mask_seq: (Batch, Seq_Len), 1 for actual items, 0 for padding
        #mask_seq = self.get_attention_mask(item_seq) # Use RecBole's mask generation
        mask_seq = (item_seq>0).float()
        
        # 4. Call DiffuRec's forward (diffusion process)
        # We need to explicitly handle the `train_flag` for DiffuRec's internal logic.
        # Inside DiffuRec, `forward` already assumes training mode, `reverse_p_sample` assumes inference.
        x_0_pred_latent, seq_rep_diffusion_model, weights, t, x_t_noisy, dy_pre_x_t, e_v_S, e_t_S, Rep_v, Rep_t = \
            self.diffu(item_rep_tuple, item_tag_tuple, mask_seq)

        # 5. Calculate Loss
        # x_0_pred_latent is the predicted target item embedding in the latent space.
        # next_item is the actual target item ID.
        
        # Option 1: Use `loss_diffu_ce` (cross-entropy on scores against all items)
        # scores = torch.matmul(x_0_pred_latent, self.token_embedding.weight.t())
        # loss = self.loss_ce(scores, next_item) # next_item should be (Batch,) for CrossEntropyLoss

        # Option 2: Use `loss_rmse` if you want to explicitly regress to the target ID embedding
        # rep_gt = self.token_embedding(next_item)
        # loss = torch.sqrt(self.loss_mse(x_0_pred_latent, rep_gt))
        
        # Let's use the cross-entropy loss as it's common for recommendation.
        # The scores are computed by dot product with all item ID embeddings.
        #scores = torch.matmul(x_0_pred_latent, self.token_embedding.weight.t())
        #loss = self.loss_ce(scores, next_item)
        
        # If using LossAwareSampler, update it here (uncomment if step_sample supports it)
        # if hasattr(self.diffu.schedule_sampler, 'update_with_all_losses'):
        #     loss_for_sampler = self.loss_ce_rec(scores, next_item).detach() # detachment for LossAwareSampler update
        #     self.diffu.schedule_sampler.update_with_all_losses(t, loss_for_sampler)

        id_seq_output = self.gather_indexes(seq_rep_diffusion_model, item_seq_len - 1)
        text_seq_output = None
        image_seq_output = None
        text_emb = None
        image_emb = None
        #text_seq_output = self.gather_indexes(text_emb, item_seq_len - 1)
        #image_seq_output = self.gather_indexes(image_emb, item_seq_len - 1)
        return id_seq_output, text_seq_output, image_seq_output, text_emb, image_emb, e_v_S, e_t_S, Rep_v, Rep_t, x_0_pred_latent
    

    def calculate_info_nce_loss(self, query, positive_key, negative_keys, temperature, batch_size):
        """
        Calculates InfoNCE loss with batch-wise negative sampling.
        Assumes positive_key and negative_keys are from the same batch as query.
        Effectively, for query[i], positive is positive_key[i], and negatives are all other negative_keys[j] where j != i.
        
        :param query: [B, D]
        :param positive_key: [B, D]
        :param negative_keys: [B, D] (usually the same tensor as positive_key for batch-wise InfoNCE)
        :param temperature: Scalar
        :return: Scalar Loss
        """
        # Normalize embeddings
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        negative_keys_normalized = F.normalize(negative_keys, dim=-1) # Normalize negatives as well

        # Positive similarity [B, 1]
        l_pos = th.sum(query * positive_key, dim=-1, keepdim=True) # [B, 1]

        # Negative similarities [B, B]
        # For query[i], calculate similarity with all negative_keys[j]
        l_neg_all = th.matmul(query, negative_keys_normalized.T) # [B, B]

        # Mask out the positive sample from the negative_keys list for each query
        # The diagonal elements of l_neg_all are query[i] vs negative_keys[i], which is the positive similarity.
        # We want to exclude this from the negative_keys pool.
        
        # Create a mask to set diagonal (self-similarity) to a very small number or -infinity
        # This ensures that query[i] is NOT considered its own negative.
        
        # A common trick is to concatenate l_pos with l_neg_all where l_neg_all already contains l_pos on its diagonal.
        # Then for each row `i`, the actual positive is `l_pos[i]`, and the negatives are `l_neg_all[i, j]` for `j != i`.

        # Let's rebuild the logits in a more standard way for batch-wise InfoNCE (NT-Xent)
        # The matrix of all similarities is [B, B] where (i,j) is sim(query[i], negative_keys[j])
        all_sims_matrix = th.matmul(query, negative_keys_normalized.T) # [B, B]

        # For each query[i], its positive is positive_key[i].
        # The diagonal elements all_sims_matrix[i, i] are exactly sim(query[i], positive_key[i]).
        # So we can extract positives from the diagonal:
        positives = all_sims_matrix.diag().view(-1, 1) # [B, 1]

        # And the negatives are all other elements in each row:
        # To get negatives, we create a mask that is True on the diagonal.
        negatives_mask = ~th.eye(batch_size, dtype=th.bool, device=query.device) # [B, B], True off-diagonal

        # Apply the mask to get only negative similarities
        # Reshape to [B, B-1] where each row contains the (B-1) negative similarities for query[i]
        negatives = all_sims_matrix[negatives_mask].view(batch_size, -1) # [B, B-1]

        # Concatenate positives and negatives for the logits
        logits = th.cat([positives, negatives], dim=-1) / temperature
        
        # The label for positive sample is 0 (the first element in the concatenated logits)
        labels = th.zeros(batch_size, dtype=th.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        id_seq_output, text_seq_output, image_seq_output, text_emb, image_emb, e_v_S, e_t_S, Rep_v, Rep_t, x_0_pred_latent = self.forward(item_seq, item_seq_len, pos_items)

        cl_loss = 0
        if self.config['contrasive_loss']:
            for i in range(len(text_emb)):  
                text_view, image_view = F.normalize(text_emb[i], dim=1), F.normalize(image_emb[i], dim=1)  
  
                pos_score = (text_view @ image_view.T) / self.temp  
                score = torch.diag(F.log_softmax(pos_score, dim=1))  
                cl_loss += -score.mean()
        
        L_MC = 0
        if self.mc_loss:
            batch_size = x_0_pred_latent.shape[0]

            # --- 1. Calculate L_diff ---
            # This depends on your original TDRec implementation of L_diff.
            # L_diff = ... # Your existing L_diff calculation

            # --- 2. Calculate L_CE ---
            # L_CE = F.cross_entropy(prediction_logits, true_item_id_labels) # Your existing L_CE calculation

            # --- 3. Calculate L_MC (Multimodal Consistency Loss) ---

            # For InfoNCE, negative_keys can often be the entire batch of relevant embeddings.
            # The calculate_info_nce_loss function (as corrected previously)
            # handles batch-wise negative sampling where positive_key[i] is excluded from negative_keys[i].

            # --- Individual Item Consistency ---
            # Query: x_0 (reconstructed ID for the next item)
            # Positive Key: e_v_S (projected visual for the next item)
            # Negative Keys Pool: all e_v_S in the batch (batch-wise negative sampling)
            l_mc_item_v = self.calculate_info_nce_loss(query=x_0_pred_latent, positive_key=e_v_S, negative_keys=e_v_S, temperature=self.temperature, batch_size=batch_size)
            l_mc_item_t = self.calculate_info_nce_loss(query=x_0_pred_latent, positive_key=e_t_S, negative_keys=e_t_S, temperature=self.temperature, batch_size=batch_size)

            # --- Sequence-level Cross-Modal Consistency ---
            # Query: x_0 (reconstructed ID for the next item)
            # Positive Key: Rep_v (aggregated visual for the historical sequence)
            # Negative Keys Pool: all Rep_v in the batch
            l_mc_seq_v = self.calculate_info_nce_loss(query=x_0_pred_latent, positive_key=Rep_v, negative_keys=Rep_v, temperature=self.temperature, batch_size=batch_size)
            l_mc_seq_t = self.calculate_info_nce_loss(query=x_0_pred_latent, positive_key=Rep_t, negative_keys=Rep_t, temperature=self.temperature, batch_size=batch_size)

            # Combine all components of L_MC
            L_MC = l_mc_item_v + l_mc_item_t + l_mc_seq_v + l_mc_seq_t


        #s_time = time.time()
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.id_embedding(pos_items)
            neg_items_emb = self.id_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            if self.config["mul_ce"]:
                test_id_emb = self.id_embedding.weight
                id_logits = torch.matmul(id_seq_output, test_id_emb.transpose(0, 1))
                id_loss = self.loss_fct(id_logits, pos_items)

                with torch.no_grad():
                    test_text_emb = self.text_in_proj(self.text_emb)
                text_logits = torch.matmul(text_seq_output, test_text_emb.transpose(0, 1))
                text_loss = self.loss_fct(text_logits, pos_items)

                with torch.no_grad():
                    test_image_emb = self.image_in_proj(self.image_emb)
                image_logits = torch.matmul(image_seq_output, test_image_emb.transpose(0, 1))
                image_loss = self.loss_fct(image_logits, pos_items)
                loss = id_loss + text_loss + image_loss

            else:
                test_item_emb = self.id_embedding.weight
                logits = torch.matmul(id_seq_output, test_item_emb.transpose(0, 1))
                loss = self.loss_fct(logits, pos_items)
        #e_time = time.time()
        #calculate_loss_time = e_time - s_time
        #print("calculate_loss time: {0}s".format(calculate_loss_time))
        return loss + cl_loss + self.lambda3 * L_MC


    @torch.no_grad()
    def predict(self, interaction):
        # 1. Extract data for prediction
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        # 2. Get multi-modal features for sequence
        #seq_id_emb, seq_text_feat, seq_img_feat = self._get_multimodal_features(item_seq)
        seq_id_emb = self.id_embedding(item_seq)
        seq_text_feat = self.text_emb[item_seq]
        seq_img_feat = self.image_emb[item_seq]
        
        # Add position embeddings
        if self.config['if_dual_pos_embed']:
            seq_id_emb = seq_id_emb + self.id_dual_pos_embed
            seq_text_feat = seq_text_feat + self.text_dual_pos_embed
            seq_img_feat = seq_img_feat + self.image_dual_pos_embed
        #position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=self.device).unsqueeze(0)
        #position_embeddings = self.position_embedding(position_ids)
        #seq_id_emb = seq_id_emb + position_embeddings
        seq_id_emb = self.LayerNorm(seq_id_emb)
        seq_id_emb = self.dropout_layer(seq_id_emb) # Dropout is usually turned off in eval, but still part of layer.
        
        item_rep_tuple = (seq_id_emb, seq_text_feat, seq_img_feat)

        # 3. Prepare sequence mask
        mask_seq = (item_seq>0).float()
        #mask_seq = self.get_attention_mask(item_seq)

        # 4. Generate initial noise for reverse process
        # The noise_x_t should have the same shape as the predicted item embedding (Batch, Hidden_Size)
        batch_size = item_seq.size(0)
        noise_x_t = th.randn(batch_size, self.hidden_size, device=self.device)

        # 5. Call DiffuRec's reverse process to get the predicted x_0 (latent embedding)
        # The `reverse` method in DiffuRec requires fused_item_rep, which means we apply modal_fusion first.
        if self.fusion_type != 'dydiff':
            fused_seq_rep = self.diffu.modal_fusion(*item_rep_tuple) # (Batch, Seq_Len, Hidden)
            predicted_latent_item_emb = self.diffu.reverse_p_sample(fused_seq_rep, noise_x_t, mask_seq) # (Batch, Hidden)
        else:
            predicted_latent_item_emb = self.diffu.reverse_p_sample_mm(item_rep_tuple, noise_x_t, mask_seq) # (Batch, Hidden)

        # 6. Calculate scores against all items' ID embeddings
        test_item_emb = self.id_embedding(test_item)
        scores = torch.matmul(predicted_latent_item_emb, test_item_emb.t()).sum(dim=1) 
        return scores

    def full_sort_predict(self, interaction):
        # For full-sort evaluation, `predict` usually handles it by scoring against all items.
        # 1. Extract data for prediction
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # 2. Get multi-modal features for sequence
        #seq_id_emb, seq_text_feat, seq_img_feat = self._get_multimodal_features(item_seq)
        seq_id_emb = self.id_embedding(item_seq)
        seq_text_feat = self.text_emb[item_seq]
        seq_img_feat = self.image_emb[item_seq]
        
        # Add position embeddings
        if self.config['if_dual_pos_embed']:
            seq_id_emb = seq_id_emb + self.id_dual_pos_embed
            seq_text_feat = seq_text_feat + self.text_dual_pos_embed
            seq_img_feat = seq_img_feat + self.image_dual_pos_embed
        #position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=self.device).unsqueeze(0)
        #position_embeddings = self.position_embedding(position_ids)
        #seq_id_emb = seq_id_emb + position_embeddings
        seq_id_emb = self.LayerNorm(seq_id_emb)
        seq_id_emb = self.dropout_layer(seq_id_emb) # Dropout is usually turned off in eval, but still part of layer.
        
        item_rep_tuple = (seq_id_emb, seq_text_feat, seq_img_feat)

        # 3. Prepare sequence mask
        mask_seq = (item_seq>0).float()
        #mask_seq = self.get_attention_mask(item_seq)

        # 4. Generate initial noise for reverse process
        # The noise_x_t should have the same shape as the predicted item embedding (Batch, Hidden_Size)
        batch_size = item_seq.size(0)
        noise_x_t = th.randn(batch_size, self.hidden_size, device=self.device)

        # 5. Call DiffuRec's reverse process to get the predicted x_0 (latent embedding)
        # The `reverse` method in DiffuRec requires fused_item_rep, which means we apply modal_fusion first.
        if self.fusion_type != 'dydiff':
            fused_seq_rep = self.diffu.modal_fusion(*item_rep_tuple) # (Batch, Seq_Len, Hidden)
            predicted_latent_item_emb = self.diffu.reverse_p_sample(fused_seq_rep, noise_x_t, mask_seq) # (Batch, Hidden)
        else:
            predicted_latent_item_emb = self.diffu.reverse_p_sample_mm(item_rep_tuple, noise_x_t, mask_seq) # (Batch, Hidden)

        # 6. Calculate scores against all items' ID embeddings
        test_item_emb = self.id_embedding.weight
        id_scores = torch.matmul(predicted_latent_item_emb, test_item_emb.transpose(0, 1))  # [B, n_items]

        if self.config["mul_eval"]:
            with torch.no_grad():
                test_text_emb = self.text_in_proj(self.text_emb)
            text_scores = torch.matmul(text_seq_output, test_text_emb.transpose(0, 1))

            with torch.no_grad():
                test_image_emb = self.image_in_proj(self.image_emb)
            image_scores = torch.matmul(image_seq_output, test_image_emb.transpose(0, 1))
            
            scores = torch.stack([id_scores, text_scores, image_scores], dim=0).mean(dim=0)
        else:
            scores = id_scores
        
        return scores
        #return self.predict(interaction)
  

      
