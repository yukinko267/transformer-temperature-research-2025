Temperature-Scaled Transformer Attention

本研究は、TransformerのAttention内Softmaxに温度パラメータを導入し、その挙動と性能変化を分析することを目的とする。
特に、Attention分布の鋭さと翻訳性能の関係に着目する。

Overview

Multi-Head Attention内のSoftmaxに温度Tを導入

学習時および評価時で温度を制御可能

Attentionの可視化およびdump機能付き

Environment

Python 3.x

PyTorch

CUDA対応GPU推奨

Usage:
python -m main.run_all
Notes

本リポジトリは研究目的で作成された実験コードである。