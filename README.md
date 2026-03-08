# Temperature-Scaled Attention Transformer

## 概要
本研究は，機械翻訳タスクにおいてTransformerの性能とAttention機構の挙動を分析することを目的とした実装研究である．特に，TransformerのAttention機構におけるSoftmax関数に温度パラメータ(temperature)を導入し，学習過程で温度を徐々に低下させる冷却スケジュール(cooling schdule)を適用する．  
これにより，Attention分布の集中度の変化の仕方や翻訳性能への影響を分析する．

## 手法
Softmax温度を制御することで，Attentionの集中度を調節する．  
温度が高い場合，Attention分布は広く分散し，温度が低い場合，Attention分布は特定のトークンに集中する．
<img width="989" height="396" alt="温度に対する分布" src="https://github.com/user-attachments/assets/62247629-9daa-4726-9be6-e19de6a0711f" />

また，Attention の挙動を定量的に評価するため，各層・各ヘッドにおけるAttention 分布の正規化エントロピーを計算する．

これにより，Attention がどの程度集中しているか／分散しているかをヘッド単位で可視化・分析する．

## データセット
学習データおよび評価データのロードにはHuggingFace Datasets を使用する．

## 温度変化

本研究では，温度関数・初期温度・最終温度を変化させた実験を行う．

## 🛠 設定方法

プロジェクトの設定は以下のディレクトリ構造に沿って管理されている．

```text
project_root/
├── config/
│   └── config.py          # メイン設定ファイル（パラメータ指定）
└── util/
    └── Temp_schedule.py   # 温度関数の例
```

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
