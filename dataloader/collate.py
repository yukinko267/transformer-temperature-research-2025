# data/collate.py
import torch

def collate_fn(batch):
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]

    # 各バッチの最大長を取得
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(s) for s in tgt_list)

    # pad_id は src[0][0] の sp.pad_id() を使って取得する
    pad_id = 0 

    src_batch = []
    tgt_batch = []

    for s in src_list:
        src_batch.append(s + [pad_id] * (max_src - len(s)))

    for t in tgt_list:
        tgt_batch.append(t + [pad_id] * (max_tgt - len(t)))

    return {
        "src": torch.tensor(src_batch, dtype=torch.long),
        "tgt": torch.tensor(tgt_batch, dtype=torch.long)
    }
