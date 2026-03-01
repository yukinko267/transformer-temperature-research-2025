# dataloader/__init__.py

from .configdata import get_tokenizer_info

# 関数を実行して情報を取得する
info = get_tokenizer_info()

# 外部から直接参照できるように変数として定義し直す
src_pad_idx = info["src_pad_idx"]
trg_pad_idx = info["trg_pad_idx"]
trg_sos_idx = info["trg_sos_idx"]
trg_eos_idx = info["trg_eos_idx"]
unk_idx     = info["unk_idx"]
vocab_size  = info["vocab_size"]
special_tokens = info["special_tokens"]