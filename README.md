# Temperature-Scaled Attention Transformer

## 概要

本研究は、TransformerにおけるAttention機構のSoftmaxに温度パラメータ（Temperature）を導入し学習時に冷却することで、
Attention分布の変化とモデル性能への影響を分析する研究実装です。

Softmaxの温度を制御することで、Attentionの集中度（鋭さ）を調整可能にし、
その挙動をヘッドごとの正規化エントロピーで可視化・定量評価します。

## 温度変化

本研究では、温度関数・初期温度・最終温度を変化させた実験を行います。

## 🛠 設定方法

プロジェクトの設定は以下のディレクトリ構造に沿って管理されています。

```text
project_root/
├── config/
│   └── config.py          # メイン設定ファイル（パラメータ指定）
└── util/
    └── Temp_schedule.py   # 温度関数の例

## 保存ディレクトリ
- saved_tokenizer_data : トークナイズ化されたモデル(.model)
- saved_model : エポックごとのモデル(.pt)
- saved_attention_data : attention内部データ(.pkl)
- saved_attention_heatmaps : attentionヘッドごとのHeatmap(.png)

## 主な特徴

- Multi-Head Attention内のSoftmaxに温度パラメータを導入
- 学習時に温度変更可能
- Attention重みの保存（dump）機能

## 実行環境

- Python 3.x
- PyTorch
- CUDA対応GPU（推奨）

## 実行方法

```bash
python -m main.run_all # 全処理
python -m main.attention_analysis # 正規化エントロピー可視化のみ