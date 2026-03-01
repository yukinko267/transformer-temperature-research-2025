import torch

def greedy_decode(model, src, trg_sos_idx, trg_eos_idx, max_len=50):
    model.eval()
    with torch.no_grad():
        batch_size = src.size(0)
        device = src.device

        # 初期トークン（SOS）
        generated = torch.full((batch_size, 1), trg_sos_idx, dtype=torch.long).to(device)

        for _ in range(max_len):
            out = model(src, generated)  # [batch, seq_len, vocab]
            next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)  # [batch, 1]

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == trg_eos_idx).all():
                break

        return generated
