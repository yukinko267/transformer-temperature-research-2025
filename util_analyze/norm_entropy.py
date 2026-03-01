import torch
import numpy as np

# ======================================================
# Per-head metric
# ======================================================

def attention_metric_per_head(attn: torch.Tensor, metric: str):
    """
    attn:
      - metric="entropy"   : [H, Q, K] (softmax後)
      - metric="logit_var" : [H, Q, K] (softmax前)
      - metric="diag"      : [H, Q, K] (softmax後)

    return: [H]
    """
    if metric == "entropy":
        eps = 1e-9
        p = attn.clamp(min=eps)

        ent = -(p * p.log()).sum(dim=-1)   # [H, Q]
        K = p.size(-1)
        ent_norm = ent / np.log(K)

        return ent_norm.mean(dim=-1)       # [H]

    elif metric == "logit_var":
        var = attn.var(dim=-1, unbiased=False)
        return var.mean(dim=-1)             # [H]

    elif metric == "diag":
        H, Q, K = attn.shape
        diag_len = min(Q, K)

        diag_sum = attn[:, torch.arange(diag_len), torch.arange(diag_len)].sum(dim=-1)
        total_sum = attn.sum(dim=(1, 2))

        return diag_sum / (total_sum + 1e-9)  # [H]

    else:
        raise ValueError(f"Unknown metric: {metric}")


# ======================================================
# JS divergence utilities
# ======================================================

def js_divergence(p, q, eps=1e-8):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return 0.5 * (kl_pm + kl_qm)


def head_attention_distribution(attn):
    """
    attn: shape (Q, K)
    return: shape (K,)
    """
    return attn.mean(axis=0)


def layer_head_js_divergence(attn_layer):
    """
    attn_layer: shape (H, Q, K) (numpy)
    return: scalar
    """
    num_heads = attn_layer.shape[0]

    head_dists = [
        head_attention_distribution(attn_layer[h])
        for h in range(num_heads)
    ]

    js_vals = []
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            js_vals.append(
                js_divergence(head_dists[i], head_dists[j])
            )

    return np.mean(js_vals)


# ======================================================
# Main interface
# ======================================================

def compute_attention_metric(
    exp_data_a,
    layer_id,
    metric,
    mode="mono",
    attn_type="cross",   # ← 追加
    exp_data_b=None,
):
    """
    metric:
      - "entropy"
      - "logit_var"
      - "diag"
      - "js_head_div"

    mode:
      - "mono"
      - "delta"

    attn_type:
      - "enc_self"
      - "dec_self"
      - "cross"

    return:
      mono  -> np.ndarray [S, H]
      delta -> np.ndarray [S, H]
    """

    sent_ids = exp_data_a["sent_id"]

    if mode == "delta":
        assert exp_data_b is not None
        assert sent_ids == exp_data_b["sent_id"]

    # ==================================================
    # JS head divergence（softmax後）
    # ==================================================
    if metric == "js_head_div":
        results = []

        attn_post = exp_data_a["attn"][attn_type]["post"]

        for sid in sent_ids:
            attn_layer = attn_post[sid][layer_id]  # [H, Q, K]
            attn_layer_np = attn_layer.cpu().numpy()

            js_val = layer_head_js_divergence(attn_layer_np)

            # heatmap互換：[H]
            results.append(
                np.full(attn_layer_np.shape[0], js_val)
            )

        return np.stack(results)  # [S, H]

    # ==================================================
    # entropy / logit_var / diag
    # ==================================================
    key = "post" if metric in ["entropy", "diag"] else "pre"

    results = []

    for sid in sent_ids:
        A = attention_metric_per_head(
            exp_data_a["attn"][attn_type][key][sid][layer_id],
            metric
        )

        if mode == "mono":
            results.append(A.cpu().numpy())

        elif mode == "delta":
            B = attention_metric_per_head(
                exp_data_b["attn"][attn_type][key][sid][layer_id],
                metric
            )
            results.append((B - A).cpu().numpy())

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return np.stack(results)  # [S, H]
