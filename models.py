import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import normalize


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# ---------------------------------------------------------------------------
# HAT+ Module 1: HierarchicalContextEncoder
# ---------------------------------------------------------------------------
# Enriches the short-window encoder output via two stages:
#   Stage 1 (ctx_encoder)  — instance-level self-attention over encoded_x
#   Stage 2 (ctx_decoder)  — compact context tokens via cross-attention
# Also produces ctx_cls: a context-level classification prediction used
# as direct auxiliary supervision (same snip_label, same focal loss).
# This provides a direct gradient signal to the context encoder every step,
# ensuring it learns to represent current activity — not just pass gradients
# through the memory chain.
# ---------------------------------------------------------------------------
class HierarchicalContextEncoder(torch.nn.Module):
    def __init__(self, opt):
        super(HierarchicalContextEncoder, self).__init__()
        n_embedding_dim   = opt["hidden_dim"]
        n_class           = opt["num_of_class"]
        self.n_ctx_tokens = opt.get("ctx_tokens", 8)
        self.ablation_mode = opt.get("ablation_mode", "full")
        self.save_attention = opt.get("save_attention", False)
        
        n_ctx_dec_head    = 4
        n_ctx_dec_layer   = 2
        dropout           = 0.3

        # We remove ctx_encoder (redundant self-attention).
        # We keep only ctx_decoder to compress encoded_x into an information-rich summary.
        self.ctx_decoder = nn.TransformerDecoder(
                                nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                           nhead=n_ctx_dec_head,
                                                           dropout=dropout,
                                                           activation='gelu'),
                                n_ctx_dec_layer,
                                nn.LayerNorm(n_embedding_dim))

        self.ctx_token = nn.Parameter(torch.zeros(self.n_ctx_tokens, 1, n_embedding_dim))

        # Context supervision head — ensures ctx_out is non-trivial and meaningful
        self.ctx_head = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 4), nn.ReLU())
        self.ctx_classifier = nn.Sequential(
            nn.Linear(self.n_ctx_tokens * n_embedding_dim // 4,
                      (self.n_ctx_tokens * n_embedding_dim // 4) // 4),
            nn.ReLU(),
            nn.Linear((self.n_ctx_tokens * n_embedding_dim // 4) // 4, n_class))

    def forward(self, encoded_x):
        # encoded_x : [short_window_size, B, D]
        
        # Ablation support: If baseline or memory_only, context is not used.
        if self.ablation_mode in ['baseline', 'memory_only']:
            ctx_out = torch.zeros(self.n_ctx_tokens, encoded_x.shape[1], encoded_x.shape[2], device=encoded_x.device)
            ctx_cls = torch.zeros(encoded_x.shape[1], self.ctx_classifier[-1].out_features, device=encoded_x.device)
            return ctx_out, ctx_cls

        # Compact context tokens via cross-attention
        ctx_token = self.ctx_token.expand(-1, encoded_x.shape[1], -1)        # [ctx_tokens, B, D]
        ctx_out   = self.ctx_decoder(ctx_token, encoded_x)                   # [ctx_tokens, B, D]

        # Context classification
        ctx_feat = self.ctx_head(ctx_out)                                    # [ctx_tokens, B, D//4]
        ctx_feat = torch.flatten(ctx_feat.permute(1, 0, 2), start_dim=1)     # [B, ctx_tokens*D//4]
        ctx_cls  = self.ctx_classifier(ctx_feat)                             # [B, n_class]

        return ctx_out, ctx_cls


# ---------------------------------------------------------------------------
# HAT+ Module 2: DualMemoryUnit  (replaces HistoryUnit)
# ---------------------------------------------------------------------------
# Three sub-components:
#   long_mem_encoder — HAT's history_encoder_block1, preserved exactly
#   short_mem_encoder — NEW: compresses ctx_out into short-term tokens
#   memory_fusion     — replaces block2; additionally gated by short-term
#                       summary so the network can selectively weight
#                       which history tokens are relevant right now
# ---------------------------------------------------------------------------
class DualMemoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super(DualMemoryUnit, self).__init__()
        self.n_feature         = opt["feat_dim"]
        n_class                = opt["num_of_class"]
        n_embedding_dim        = opt["hidden_dim"]
        n_hist_dec_head        = 4
        n_hist_dec_layer       = 5
        n_fusion_dec_head      = 4
        n_fusion_dec_layer     = 2
        self.anchors           = opt["anchors"]
        self.history_tokens    = 16
        self.ablation_mode     = opt.get("ablation_mode", "full")
        self.save_attention    = opt.get("save_attention", False)
        dropout                = 0.3

        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)

        # Long-term memory compressor (renamed conceptually, but keep variable name to avoid massive checkpoint breaks, though we rename the component in thought)
        self.history_compressor = nn.TransformerDecoder(
                                    nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                               nhead=n_hist_dec_head,
                                                               dropout=dropout,
                                                               activation='gelu'),
                                    n_hist_dec_layer,
                                    nn.LayerNorm(n_embedding_dim))

        # Memory fusion uses ctx_out directly (no short_mem_encoder)
        self.memory_fusion = nn.TransformerDecoder(
                                    nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                               nhead=n_fusion_dec_head,
                                                               dropout=dropout,
                                                               activation='gelu'),
                                    n_fusion_dec_layer,
                                    nn.LayerNorm(n_embedding_dim))

        # Gated memory conditioned on context summary
        self.mem_gate = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(n_embedding_dim // 4, n_embedding_dim),
            nn.Sigmoid())

        self.snip_head = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 4), nn.ReLU())
        self.snip_classifier = nn.Sequential(
            nn.Linear(self.history_tokens * n_embedding_dim // 4,
                      (self.history_tokens * n_embedding_dim // 4) // 4),
            nn.ReLU(),
            nn.Linear((self.history_tokens * n_embedding_dim // 4) // 4, n_class))

        self.history_token   = nn.Parameter(torch.zeros(self.history_tokens,   1, n_embedding_dim))

        self.norm2    = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, long_x, ctx_out):
        # long_x  : [48, B, D]
        # ctx_out : [ctx_tokens, B, D]

        if self.ablation_mode == 'context_only':
            # In context_only mode, we don't process history at all
            dummy_mem = torch.zeros(self.history_tokens, long_x.shape[1], long_x.shape[2], device=long_x.device)
            dummy_cls = torch.zeros(long_x.shape[1], self.snip_classifier[-1].out_features, device=long_x.device)
            return dummy_mem, dummy_cls

        hist_pe_x     = self.history_positional_encoding(long_x)
        history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)
        long_mem      = self.history_compressor(history_token, hist_pe_x)    # [16, B, D]

        # Snippet Classification Head on pure history
        snippet_feat = self.snip_head(long_mem)
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        snip_cls     = self.snip_classifier(snippet_feat)

        if self.ablation_mode in ['baseline', 'memory_only']:
            return long_mem, snip_cls

        # Gated scaling of long_mem based on context
        ctx_summary    = ctx_out.mean(dim=0)             # [B, D]
        gate           = self.mem_gate(ctx_summary)      # [B, D]
        gate           = gate.unsqueeze(0)               # [1, B, D]
        long_mem_gated = long_mem * gate                 # [16, B, D]

        # Memory Fusion: Queries = gated history, Keys/Values = context
        fused_mem = self.memory_fusion(long_mem_gated, ctx_out)  # [16, B, D]
        fused_mem = fused_mem + self.dropout2(long_mem)          # residual
        fused_mem = self.norm2(fused_mem)

        return fused_mem, snip_cls


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature      = opt["feat_dim"]
        n_class             = opt["num_of_class"]
        n_embedding_dim     = opt["hidden_dim"]
        n_enc_layer         = opt["enc_layer"]
        n_enc_head          = opt["enc_head"]
        n_dec_layer         = opt["dec_layer"]
        n_dec_head          = opt["dec_head"]
        n_comb_dec_head     = 4
        n_comb_dec_layer    = 5
        n_seglen            = opt["segment_size"]
        self.anchors        = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride = []
        self.ablation_mode  = opt.get("ablation_mode", "full")
        dropout             = 0.3
        self.best_loss      = 1000000
        self.best_map       = 0

        self.feature_reduction_rgb  = nn.Linear(self.n_feature // 2, n_embedding_dim // 2)
        self.feature_reduction_flow = nn.Linear(self.n_feature // 2, n_embedding_dim // 2)

        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)

        self.encoder = nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=n_embedding_dim,
                                                       nhead=n_enc_head,
                                                       dropout=dropout,
                                                       activation='gelu'),
                            n_enc_layer,
                            nn.LayerNorm(n_embedding_dim))

        self.decoder = nn.TransformerDecoder(
                            nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                       nhead=n_dec_head,
                                                       dropout=dropout,
                                                       activation='gelu'),
                            n_dec_layer,
                            nn.LayerNorm(n_embedding_dim))

        self.context_encoder  = HierarchicalContextEncoder(opt)
        self.dual_memory_unit = DualMemoryUnit(opt)

        # Single unified anchor refinement stage
        self.anchor_refinement_block = nn.TransformerDecoder(
                            nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                       nhead=n_comb_dec_head,
                                                       dropout=dropout,
                                                       activation='gelu'),
                            n_comb_dec_layer,
                            nn.LayerNorm(n_embedding_dim))

        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(),
            nn.Linear(n_embedding_dim, n_class))
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(),
            nn.Linear(n_embedding_dim, 2))

        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))

        self.norm_refine    = nn.LayerNorm(n_embedding_dim)
        self.dropout_refine = nn.Dropout(0.1)

    def forward(self, inputs):
        base_x_rgb  = self.feature_reduction_rgb(inputs[:, :, :self.n_feature // 2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:, :, self.n_feature // 2:].float())
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        base_x = base_x.permute([1, 0, 2])  # [T, B, D]

        short_x = base_x[-self.short_window_size:]   # [16, B, D]
        long_x  = base_x[:-self.short_window_size]   # [48, B, D]

        ## Anchor Feature Generator
        pe_x          = self.positional_encoding(short_x)
        encoded_x     = self.encoder(pe_x)
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        decoded_x     = self.decoder(decoder_token, encoded_x)

        ## Context and Memory Modules
        ctx_out, ctx_cls = self.context_encoder(encoded_x)
        hist_encoded_x, snip_cls = self.dual_memory_unit(long_x, ctx_out)

        ## Ablation Routing for Anchor Refinement
        if self.ablation_mode == 'context_only':
            memory_for_anchor = ctx_out
        else:
            memory_for_anchor = hist_encoded_x

        ## Unified Anchor Refinement
        after_refinement = self.anchor_refinement_block(decoded_x, memory_for_anchor)
        after_refinement = after_refinement + self.dropout_refine(decoded_x)
        after_refinement = self.norm_refine(after_refinement)
        
        decoded_anchor_feat = after_refinement.permute([1, 0, 2])  # [B, n_anchors, D]

        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)

        return anc_cls, anc_reg, snip_cls, ctx_cls


class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class         = opt["num_of_class"] - 1
        n_seglen        = opt["segment_size"]
        n_embedding_dim = 2 * n_seglen
        dropout         = 0.3
        self.best_loss  = 1000000
        self.best_map   = 0

        self.mlp1    = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2    = nn.Linear(n_embedding_dim, 1)
        self.norm    = nn.InstanceNorm1d(n_class)
        self.relu    = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        base_x = inputs.permute([0, 2, 1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        return x
