# src/experiments/build.py

import torch
from models.model.transformer import Transformer
from dataloader import src_pad_idx, trg_pad_idx, trg_sos_idx, vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config):
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        enc_voc_size=vocab_size,
        dec_voc_size=vocab_size,
        d_model=config.d_model,
        n_head=config.n_head,
        max_len=config.max_len,
        ffn_hidden=config.ffn_hidden,
        n_layers=config.n_layers,
        drop_prob=config.drop_prob,
        device=device
    ).to(device)

    return model, device