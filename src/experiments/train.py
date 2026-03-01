# src/experiments/train.py

import math
import time
from util.Temp_schedule import get_temperature


def train_one_epoch(model, loader, optimizer, criterion, config, device, global_step):

    max_train_steps = config.train_limit_data // config.batch_size * config.epochs

    model.train()
    epoch_loss = 0

    first10_times = []   # ★ 初回10step用
    printed_10step = False
    steps = 0

    for batch in loader:
        start = time.time()

        T = get_temperature(
            global_step,
            max_train_steps,
            config.start_Temp,
            config.end_Temp,
            config.schedule
        )

        src = batch["src"].to(device)
        trg = batch["tgt"].to(device)

        optimizer.zero_grad()

        output = model(src, trg[:, :-1], T)
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg_y = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg_y)
        loss.backward()
        optimizer.step()

        # ---- 最初の10ステップ計測 ----
        step_time = time.time() - start
        steps += 1
        if steps <= 10:
            first10_times.append(step_time)
            if steps == 10 and not printed_10step:
                avg10 = sum(first10_times) / 10
                print(f"[STEP TIME] avg over first 10 steps = {avg10:.4f} sec")
                printed_10step = True

        epoch_loss += loss.item()
        global_step += 1

    avg_loss = epoch_loss / len(loader)
    ppl = math.exp(avg_loss)

    return avg_loss, ppl, global_step