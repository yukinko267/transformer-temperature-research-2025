# data/dataloader.py
from torch.utils.data import DataLoader
from .wmt_dataset import WMTDataset
from .collate import collate_fn
from config.config import config

def create_dataloader(
    split,
    batch_size,
    max_len,
    sp_model_path=None,  # ★ デフォルトを None に変更
    shuffle=True,
    limit=None,
    num_workers=4
):
    # ★ ここでインスタンス化して、パスを取得する
    if sp_model_path is None:
        cfg = config()  # ここで __init__ が走り、SPM_MODEL_PATH が生成される
        sp_model_path = cfg.SPM_MODEL_PATH

    ds = WMTDataset(
        split=split,
        sp_model_path=sp_model_path,
        max_len=max_len,
        limit=limit,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader