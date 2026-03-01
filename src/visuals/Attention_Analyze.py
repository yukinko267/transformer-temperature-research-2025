import numpy as np
import matplotlib.pyplot as plt
import os
from util_analyze import load_attn_pkls, compute_attention_metric

def generate_attention_heatmaps(
    start_Temp: float,
    end_Temp: float,
    schedule: str,
    seed: int,
    epochs: int
):
    """
    指定された実験条件の.pklファイルを読み込み、ヘッドごとの正規化エントロピーを可視化する。
    """
    base_path = "saved_attention_data"
    save_dir = "saved_attention_heatmaps"
    attn_types = ["enc_self", "dec_self", "cross"]

    os.makedirs(save_dir, exist_ok=True)

    for attn_type in attn_types:
        # 1. PKLパスの生成 (命名規則: attention_{start_Temp}_{end_Temp}_{schedule}_{seed}_{e}.pkl)
        pkl_paths = [
            f"{base_path}/attention_{start_Temp}_{end_Temp}_{schedule}_{seed}_{e}.pkl"
            for e in range(1, epochs + 1)
        ]
        
        # ファイルの存在確認（エラー防止）
        valid_paths = [p for p in pkl_paths if os.path.exists(p)]
        if not valid_paths:
            print(f"Warning: No files found for {attn_type} in {start_Temp}_{end_Temp}_{schedule}_{seed}. Skipping...")
            continue

        epochs_data = load_attn_pkls(valid_paths)
        num_layers = epochs_data[0]["num_layers"]

        # 2. Figureの初期化
        fig, axes = plt.subplots(
            1, len(epochs_data), 
            figsize=(4 * len(epochs_data), 6), 
            sharey=True
        )
        # エポック1つの場合にaxesが配列でなくなるのを防ぐ
        if len(epochs_data) == 1: axes = [axes]

        valid_epochs = [e for e in range(1, epochs + 1) if os.path.exists(f"{base_path}/attention_{start_Temp}_{end_Temp}_{schedule}_{seed}_{e}.pkl")]

        for ax, epoch_id, exp_data in zip(axes, valid_epochs, epochs_data):
            layer_head_values = []

            for layer_id in range(num_layers):
                # 正規化エントロピーを計算
                metric_values = compute_attention_metric(
                    exp_data,
                    layer_id=layer_id,
                    metric="entropy", 
                    attn_type=attn_type
                )
                # 文章間平均をとって [num_head] を取得
                layer_head_values.append(metric_values.mean(axis=0))

            # [Layers, Heads] 行列に変換
            matrix = np.stack(layer_head_values)

            # 3. ヒートマップ描画
            im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect='auto')
            
            # 各パネルの装飾
            ax.set_title(f"Epoch {epoch_id}")
            ax.set_xlabel(f"Head\n(mean={matrix.mean():.3f})")
            ax.set_ylabel("Layer" if ax == axes[0] else "")
            
            # 目盛り設定 (Layer 0, 1, 2...)
            ax.set_yticks(range(num_layers))
            ax.set_xticks(range(matrix.shape[1]))

        # 4. 全体仕上げ
        fig.colorbar(im, ax=axes, shrink=0.7, label="Normalized Entropy")
        fig.suptitle(f"Research: {start_Temp}_{end_Temp}_{schedule}_{seed} | Type: {attn_type}", fontsize=14)
        
        plt.tight_layout()
        save_path = f"{save_dir}/entropy_{attn_type}_{start_Temp}_{end_Temp}_{schedule}_{seed}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Successfully saved heatmap: {save_path}")
        plt.close()

# --- 使い方例 ---
if __name__ == "__main__":
    generate_attention_heatmaps(
        start_Temp=10.0,
        end_Temp=0.1,
        schedule="linear",
        seed=43,
        epochs=5
    )