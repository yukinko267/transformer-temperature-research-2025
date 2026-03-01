import sentencepiece as spm
from config.config import config

# グローバルに置かず、関数にする
def get_tokenizer_info():
    cfg = config()
    sp = spm.SentencePieceProcessor()
    
    # ここで初めてロードを試みる
    if not sp.load(cfg.SPM_MODEL_PATH):
        raise FileNotFoundError(f"Tokenizer model not found at {cfg.SPM_MODEL_PATH}. "
                                "Make sure build_sentencepiece_tokenizer() has run.")

    info = {
        "src_pad_idx": sp.pad_id(),
        "trg_pad_idx": sp.pad_id(),
        "trg_sos_idx": sp.bos_id(),
        "trg_eos_idx": sp.eos_id(),
        "unk_idx": sp.unk_id(),
        "vocab_size": sp.get_piece_size(),
        "special_tokens": [sp.pad_id(), sp.bos_id(), sp.eos_id(), sp.unk_id()]
    }
    return info