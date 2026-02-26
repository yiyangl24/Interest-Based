import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from tqdm import  tqdm

class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs


class Contrastive_loss2(nn.Module):
    ''' copy from LLM4CDSR SIGIR25'''
    def __init__(self, tau=1) -> None:
        super().__init__()
        self.temperature = tau

    def forward(self, X, Y):
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax((X_similarity + Y_similarity) / 2 * self.temperature, dim=-1)
        X_loss = self.cross_entropy(logits, targets)
        Y_loss = self.cross_entropy(logits.T, targets.T)
        loss = (Y_loss + X_loss) / 2.0
        return loss.mean()

    def cross_entropy(self, y_pred, y_true, reduction="none"):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (- y_true * log_softmax(y_pred)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


class InterestBlock(nn.Module):

    def __init__(self, hidden_dim, session_cnt, dropout=0.3):
        super().__init__()
        self.interest_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.layernorm = nn.LayerNorm(hidden_dim, eps=1e-8)

        self.scaled = nn.Parameter(torch.tensor(0.1))

    def forward(self, seq, session_tensor, session_mask):
        """
        seq: [B, T, D]
        session_tensor: [B, S, D]
        session_mask:   [B, S]
        padding_mask:   [B, T]
        """
        seq = self.layernorm(seq)   # Pre-LN

        session_proj = self.interest_weight(session_tensor)

        session_norm = F.normalize(session_proj, dim=-1)
        feats_norm = F.normalize(seq, dim=-1)

        scores = feats_norm @ session_norm.transpose(1, 2)
        scores = scores.masked_fill(~session_mask.unsqueeze(1), -1e9)

        alpha = torch.softmax(scores, dim=-1)
        interest = alpha @ session_tensor

        fuse_input = torch.cat([seq, interest, seq - interest, seq * interest], dim=-1)

        fused = self.fuse(fuse_input)

        return self.scaled * fused


class SASRec(nn.Module):

    def __init__(self, user_cnt, item_cnt, args):

        super(SASRec, self).__init__()

        self.user_cnt = user_cnt
        self.item_cnt = item_cnt
        self.device = args.device
        self.profile = args.profile
        self.session = args.session
        self.llm_units = args.llm_units
        self.session_cnt = args.session_cnt

        self.item_emb = nn.Embedding(self.item_cnt + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_len + 1, args.hidden_units, padding_idx=0)
        self.dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):

            attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(attn_layernorm)

            attn_layer = nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(attn_layer)

            fwd_layernrom = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(fwd_layernrom)

            fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(fwd_layer)

        if args.profile:
            self.llm_user_emb = nn.Embedding(self.user_cnt + 1, args.llm_units, padding_idx=0)
            self.llm_user_adapter = nn.Sequential(
                nn.Linear(args.llm_units, args.llm_units // 2),
                nn.LayerNorm(args.llm_units // 2),
                nn.LeakyReLU(),
                nn.Linear(args.llm_units // 2, args.hidden_units)
            )
            self.user_loss_func = Contrastive_loss2(tau=args.tau)

        if args.session:
            self.llm_session_adapter = nn.Sequential(
                nn.Linear(args.llm_units, args.llm_units // 2),
                nn.LayerNorm(args.llm_units // 2),
                nn.LeakyReLU(),
                nn.Linear(args.llm_units // 2, args.hidden_units)
            )

            self.interest_block = nn.ModuleList([
                InterestBlock(args.hidden_units, args.session_cnt)
                for _ in range(args.num_blocks)
            ])

        self.init_weights(args)

    def init_weights(self, args):

        for name, param in self.named_parameters():
            try:
                nn.init.xavier_normal_(param.data)
            except:
                pass
        self.pos_emb.weight.data[0, :] = 0
        self.item_emb.weight.data[0, :] = 0

        processd_dir = f'./data/processed/{args.dataname}/'

        if args.llm_init:
            with open(processd_dir + 'item_pca_embedding.pkl', 'rb') as f:
                pca_embedding = pickle.load(f)
            pca_embedding = np.array(pca_embedding)
            pca_embedding = torch.tensor(pca_embedding, dtype=self.item_emb.weight.dtype, device=args.device)
            self.item_emb.weight.data[1:].copy_(pca_embedding)

        if args.profile:
            with open(processd_dir + 'profile_embedding.pkl', 'rb') as f:
                user_embedding = pickle.load(f)
            user_embedding = np.array(user_embedding)
            user_embedding = torch.tensor(user_embedding, dtype=self.llm_user_emb.weight.dtype, device=args.device)
            self.llm_user_emb.weight.data[0, :] = 0
            self.llm_user_emb.weight.data[1:].copy_(user_embedding)
            self.llm_user_emb.weight.requires_grad = False

        if args.session:
            with open(processd_dir + "session_embedding.pkl", 'rb') as f:
                all_session_uid, all_session_sid, all_session_embedding = pickle.load(f)

            self.session_tensor = torch.zeros(self.user_cnt + 1, self.session_cnt, self.llm_units, device=self.device)
            self.session_mask = torch.zeros(self.user_cnt + 1, self.session_cnt, device=self.device, dtype=torch.bool)

            for uid, sid, s_emb in tqdm(zip(all_session_uid, all_session_sid, all_session_embedding)):
                self.session_tensor[uid, sid] = torch.tensor(s_emb, device=self.device)
                self.session_mask[uid, sid] = True

            for block in self.interest_block:
                block.scaled.data.zero_()



    def seq2feats(self, seq, uid):

        padding_mask = torch.BoolTensor(seq == 0).to(self.device)

        positions = np.tile(np.arange(1, seq.shape[1] + 1), [seq.shape[0], 1])
        positions *= (seq != 0)

        seq = self.item_emb(torch.LongTensor(seq).to(self.device))
        seq *= self.item_emb.embedding_dim ** 0.5
        seq += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seq = self.dropout(seq)
        seq *= ~padding_mask.unsqueeze(-1)

        seq_len = seq.shape[1]
        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))

        if self.session:
            uid = torch.LongTensor(uid).to(self.device)
            session_tensor = self.session_tensor[uid]
            session_mask = self.session_mask[uid]
            session_tensor = self.llm_session_adapter(session_tensor)

        for i in range(len(self.attention_layers)):

            if self.session:
                seq = seq + self.interest_block[i](seq, session_tensor, session_mask)
                seq *= ~ padding_mask.unsqueeze(-1)

            seq = seq.transpose(0, 1)
            Q = self.attention_layernorms[i](seq)
            mha_outputs, _ = self.attention_layers[i](Q, seq, seq, attn_mask=attention_mask)
            seq = Q + mha_outputs
            seq = seq.transpose(0, 1)
            seq = seq + self.forward_layers[i](self.forward_layernorms[i](seq))
            seq *= ~padding_mask.unsqueeze(-1)

        seq = self.last_layernorm(seq)

        return seq

    def forward(self, uid, seq, pos, neg):

        feats = self.seq2feats(seq)     # [B, T, D]

        pos_embs = self.item_emb(torch.LongTensor(pos).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg).to(self.device))

        pos_logits = (feats * pos_embs).sum(dim=-1)
        neg_logits = (feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def contrastive_loss(self, uid, seq, pos, neg):

        feats = self.seq2feats(seq)     # 用 user profile 对 base 用户表征进行正则

        user_embs = feats[:, -1, :]
        llm_user_embs = self.llm_user_adapter(self.llm_user_emb(torch.LongTensor(uid).to(self.device)))

        return self.user_loss_func(llm_user_embs, user_embs)


    def predict(self, uid, seq, candidate):

        feats = self.seq2feats(seq)

        user_representation = feats[:, -1, :]       # [B, D]

        candidate_embs = self.item_emb(torch.LongTensor(candidate).to(self.device)) # [B, C, D]
        logits = candidate_embs.matmul(user_representation.unsqueeze(-1)).squeeze(-1)

        return logits

