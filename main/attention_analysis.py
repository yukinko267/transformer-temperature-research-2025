"""
Attention_Analyze.py
単体で実行可能：
指定された実験条件の.pklファイルを読み込み、ヘッドごとの正規化エントロピーを可視化
Usage:
    python -m 
"""


from src.visuals.Attention_Analyze import generate_attention_heatmaps

cfg = {
    "start_Temp": 0.5,
    "end_Temp": 2.0,
    "schedule": "linear",
    "seed": 42,
    "epochs": 5
}

generate_attention_heatmaps(
        start_Temp=cfg.start_Temp,
        end_Temp=cfg.end_Temp,
        schedule=cfg.schedule,
        seed=cfg.seed,
        epochs=cfg.epochs
    )