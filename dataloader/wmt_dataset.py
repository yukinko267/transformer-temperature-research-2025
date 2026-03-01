# data/wmt14_dataset.py
import sentencepiece as spm
from datasets import load_dataset
from torch.utils.data import Dataset
from config.config import config as ConfigClass  # 名前が被らないようにrename

class WMTDataset(Dataset):
    def __init__(self, split, sp_model_path, max_len, limit=None):
        self.cfg = ConfigClass() 
        
        # インスタンスの属性を使ってデータセットをロード
        ds = load_dataset(self.cfg.wmt, self.cfg.ts)[split]

        if limit is not None:
            limit = min(limit, len(ds))
            ds = ds.select(range(limit))

        self.ds = ds
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

        self.max_len = max_len
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def encode(self, text):
        ids = self.sp.encode(text, out_type=int)
        ids = ids[: self.max_len - 2]
        return [self.bos_id] + ids + [self.eos_id]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]["translation"]
        # self.cfg から言語設定を取得
        src_text = item[self.cfg.source]
        tgt_text = item[self.cfg.target]

        src_ids = self.encode(src_text)
        tgt_ids = self.encode(tgt_text)

        return {"src": src_ids, "tgt": tgt_ids}