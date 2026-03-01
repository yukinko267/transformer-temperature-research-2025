"""
build_tokenizer.py

WMT データから SentencePiece (BPE) の shared vocabulary を生成するモジュール。

Usage:
    from tokenizer.build_tokenizer import build_sentencepiece_tokenizer

    build_sentencepiece_tokenizer(
        source="en",
        target="cs",
        wmt_name="wmt19",
        vocab_size=16000
    )
"""

import os
from datasets import load_dataset
import sentencepiece as spm


def build_sentencepiece_tokenizer(
    source: str,
    target: str,
    wmt_name: str,
    vocab_size: int,
):
    """
    SentencePiece tokenizer を学習する関数

    Args:
        source (str): source language (例: "en")
        target (str): target language (例: "cs")
        wmt_name (str): WMT dataset name (例: "wmt19")
        vocab_size (int): 語彙サイズ
        output_dir (str): 保存先ディレクトリ（Noneならこのファイルの場所）
    """

    output_dir = "saved_tokenizer_data" # tokenizerディレクトリに保存

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    spm_txt = f"spm_input_{source}{target}.txt"
    model_prefix_name = f"spm{vocab_size}_{source}{target}"

    input_txt = os.path.join(output_dir, spm_txt)
    model_prefix = os.path.join(output_dir, model_prefix_name)

    dataset_key = f"{target}-{source}"

    # ============================================
    # 1. データ準備
    # ============================================
    if not os.path.exists(input_txt):
        print("==> Loading dataset from HuggingFace...")
        ds = load_dataset(wmt_name, dataset_key)["train"]

        print("==> Writing raw text to:", input_txt)
        with open(input_txt, "w", encoding="utf-8") as f:
            for item in ds:
                src = item["translation"][source].replace("\n", " ")
                tgt = item["translation"][target].replace("\n", " ")
                f.write(src + "\n")
                f.write(tgt + "\n")

        print("✓ Raw text file created.")
    else:
        print("==> Reusing existing raw text file.")


    # ============================================
    # 2. SentencePiece 学習
    # ============================================
    model_file = f"{model_prefix}.model"
    
    # すでにモデルファイルが存在する場合は学習をスキップ
    if os.path.exists(model_file):
        print(f"==> Reusing existing tokenizer model: {model_file}")
    else:
        print(f"==> Training SentencePiece (BPE, vocab={vocab_size})...")
        spm.SentencePieceTrainer.Train(
            f"--input={input_txt} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--model_type=bpe "
            f"--character_coverage=1.0 "
            f"--max_sentence_length=999999 "
            f"--pad_id=0 --pad_piece=<pad> "
            f"--unk_id=1 --unk_piece=<unk> "
            f"--bos_id=2 --bos_piece=<bos> "
            f"--eos_id=3 --eos_piece=<eos>"
        )
        print("✓ Tokenizer training complete!")

    print("Generated/Existing files:")
    print(f" - {model_file}")
    print(f" - {model_prefix}.vocab")

    return model_file