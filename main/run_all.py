# main/run_all.py
"""
main/run_all.py
すべての処理を実行：
toknizerの構築 → モデルの学習 → Attentionの可視化
Usage:
    python -m main.run_all
"""

from config.config import config
from src.tokenizer.build_tokenizer import build_sentencepiece_tokenizer
from src.visuals.Attention_Analyze import generate_attention_heatmaps

if __name__ == "__main__":
    cfg = config() # インスタンス化

    # 先にトークナイザーを作る
    build_sentencepiece_tokenizer(
        source=cfg.source,
        target=cfg.target,
        wmt_name=cfg.wmt,
        vocab_size=cfg.vocab_size
    )
    
    from src.experiments.run import run # 循環importを避ける(SPM_MODEL_PATH)
    run(cfg)

    #attention analyze
    generate_attention_heatmaps(
        start_Temp=cfg.start_Temp,
        end_Temp=cfg.end_Temp,
        schedule=cfg.schedule,
        seed=cfg.seed,
        epochs=cfg.epochs
    )