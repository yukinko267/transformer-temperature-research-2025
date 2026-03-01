import torch
import torch.nn.functional as F
from config.config import config

@torch.no_grad()
def beam_search_lp(
    model,
    src,
    sos_id,
    eos_id,
    src_pad_idx,
    beam_size=4,
    alpha=0.6, 
    T=1.0
):
    device = src.device
    cfg = config()


    # ---- Encoder ----
    src_mask = model.make_src_mask(src)
    enc_out = model.encoder(src, src_mask, T)  # [1, src_len, d_model]

    # expand → beam_size 個の同一エンコードを作る（並列処理用）
    enc_out = enc_out.expand(beam_size, -1, -1)
    src_mask = src_mask.expand(beam_size, 1, 1, -1)

    # ---- Beam 初期化 ----
    sequences = torch.full((beam_size, 1), sos_id, device=device, dtype=torch.long)
    scores = torch.zeros(beam_size, device=device)  # 各ビームの log_prob
    finished = []

    src_len = src.size(1)
    raw_max_len = src_len + 50
    generate_max_len = min(raw_max_len, cfg.max_len)

    for step in range(generate_max_len):

        # 全ビームのseqをまとめて decoder にぶん投げる
        tgt_mask = model.make_trg_mask(sequences)
        dec_out = model.decoder(sequences, enc_out, tgt_mask, src_mask, T)
        logits = dec_out[:, -1, :]                     # [beam, vocab]
        log_probs = F.log_softmax(logits, dim=-1)      # [beam, vocab]

        # beam 内 topK（beam_size 個の候補それぞれで topK）
        top_log_probs, top_ids = torch.topk(log_probs, beam_size, dim=-1)

        # 次の候補をまとめて構築
        candidates = []

        for i in range(beam_size):
            for k in range(beam_size):
                new_seq = torch.cat([
                    sequences[i],
                    top_ids[i, k].unsqueeze(0)
                ])
                new_score = scores[i] + top_log_probs[i, k]
                candidates.append((new_score, new_seq))

        # ソートして beam_size 個に絞る
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

        # 更新
        scores = torch.stack([b[0] for b in beams])
        sequences = torch.stack([b[1] for b in beams])

        # EOS で終了したビームの処理
        alive = []
        for score, seq in zip(scores, sequences):
            if seq[-1].item() == eos_id:
                finished.append((score, seq))
            else:
                alive.append((score, seq))

        if len(alive) == 0:
            break  # 全部 EOS なら終わり

        # 生きてるビームだけに絞る（EOS に到達したビームは finished に入れっぱなし）
        scores = torch.stack([s for s, _ in alive])
        sequences = torch.stack([seq for _, seq in alive])

        # enc_out / mask も alive に合わせてダウンサイズ
        enc_out = enc_out[:len(alive)]
        src_mask = src_mask[:len(alive)]
        beam_size = len(alive)

    # ---- 最終スコア (length penalty) ----
    candidates = finished + list(zip(scores, sequences))

    def lp_score(log_prob, seq):
        L = len(seq)
        lp = ((5 + L) ** alpha) / ((5 + 1) ** alpha)
        return log_prob / lp

    best = max(candidates, key=lambda x: lp_score(x[0], x[1]))
    return best[1]
