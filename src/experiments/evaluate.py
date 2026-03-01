import math
import torch
import pickle
import os
from util.bleu import compute_bleu
from util.beam_search import beam_search_lp
from util.Temp_schedule import get_temperature
from dataloader import src_pad_idx, trg_pad_idx, trg_sos_idx, trg_eos_idx, special_tokens


def evaluate(
    model,
    loader,
    criterion,
    config,
    device,
    global_step,
    dump_attention=False,
    save_path_file=None,
    num_sentences=None
):
    
    max_train_steps = config.train_limit_data // config.batch_size * config.epochs

    model.eval()
    epoch_loss = 0
    bleus = []

    # -------- dump buffer --------
    if dump_attention:
        dump = {
            "run_name": config.name,
            "num_layers": len(model.decoder.layers),
            "sent_id": [],
            "sent_nll": [],
            "attn": {
                "enc_self": {"pre": {}, "post": {}},
                "dec_self": {"pre": {}, "post": {}},
                "cross": {"pre": {}, "post": {}},
            }
        }

        loss_fn_sum = torch.nn.CrossEntropyLoss(
            ignore_index=trg_pad_idx,
            reduction="sum"
        )

    with torch.no_grad():
        for idx, batch in enumerate(loader):

            if num_sentences and idx >= num_sentences:
                break

            src = batch["src"].to(device)
            trg = batch["tgt"].to(device)

            T = get_temperature(
                global_step,
                max_train_steps,
                config.start_Temp,
                config.end_Temp,
                config.schedule
            )

            # バッチ全体のLoss計算（これは効率のため一括で行う）
            output = model(src, trg[:, :-1], T)
            output_dim = output.shape[-1]

            output_flat = output.contiguous().view(-1, output_dim)
            trg_y = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_flat, trg_y)
            epoch_loss += loss.item()

            # ---- BLEU (バッチ内の最初の1行に対して1行ずつ計算するように修正) ----
            # beam_search_lp が単一文を期待しているため、バッチの0番目のみを抽出
            single_src = src[0:1] # [1, seq_len]
            
            pred = beam_search_lp(
                model,
                single_src,
                sos_id=trg_sos_idx,
                eos_id=trg_eos_idx,
                src_pad_idx=src_pad_idx,
                beam_size=4,
                alpha=0.6,
                T=T
            )

            pred_list = pred.tolist()
            ref_list = trg[0].tolist()

            # 特殊トークンを除外してBLEU計算
            ref_clean = [t for t in ref_list if t not in special_tokens]
            pred_clean = [t for t in pred_list if t not in special_tokens]

            bleus.append(compute_bleu(pred_clean, ref_clean))

            # ---- attention dump (0番目の文章の重みを保存) ----
            if dump_attention:
                vocab_size = output.size(-1)
                # 0番目の文章のNLLを計算
                nll = loss_fn_sum(
                    output[0:1].reshape(-1, vocab_size),
                    trg[0:1, 1:].reshape(-1)
                ).item()

                sent_id = idx
                dump["sent_id"].append(sent_id)
                dump["sent_nll"].append(nll)

                for key in ["enc_self", "dec_self", "cross"]:
                    dump["attn"][key]["pre"][sent_id] = {}
                    dump["attn"][key]["post"][sent_id] = {}

                for layer_id in range(len(model.decoder.layers)):

                    enc_attn = model.encoder.layers[layer_id].attention.attention
                    dec_attn = model.decoder.layers[layer_id].self_attention.attention
                    cross_attn = model.decoder.layers[layer_id].enc_dec_attention.attention

                    # すべて [0] 番目のヘッドスコアを保存（現状のロジックを維持）
                    dump["attn"]["enc_self"]["pre"][sent_id][layer_id] = \
                        enc_attn.last_score_pre[0].cpu()
                    dump["attn"]["enc_self"]["post"][sent_id][layer_id] = \
                        enc_attn.last_score_post[0].cpu()

                    dump["attn"]["dec_self"]["pre"][sent_id][layer_id] = \
                        dec_attn.last_score_pre[0].cpu()
                    dump["attn"]["dec_self"]["post"][sent_id][layer_id] = \
                        dec_attn.last_score_post[0].cpu()

                    dump["attn"]["cross"]["pre"][sent_id][layer_id] = \
                        cross_attn.last_score_pre[0].cpu()
                    dump["attn"]["cross"]["post"][sent_id][layer_id] = \
                        cross_attn.last_score_post[0].cpu()

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx+1} sentences...")

    avg_loss = epoch_loss / len(loader)
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    bleu = sum(bleus) / len(bleus) if bleus else 0.0

    # -------- save dump --------
    if dump_attention and save_path_file:
        os.makedirs(os.path.dirname(save_path_file), exist_ok=True)
        with open(save_path_file, "wb") as f:
            pickle.dump(dump, f)
        print(f"\nSaved attention dump to {save_path_file}")

    return avg_loss, ppl, bleu