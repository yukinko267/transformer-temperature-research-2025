import torch
import torch.nn as nn
from torch.optim import Adam
import time

from dataloader.dataloader import create_dataloader
from dataloader import trg_pad_idx
from src.experiments.build import build_model
from src.experiments.train import train_one_epoch
from src.experiments.evaluate import evaluate
from util.Noam_optimizer import NoamOpt


def run(config):

    global_step = 0

    print("==> Loading datasets and creating DataLoaders...")
    train_loader = create_dataloader(
        "train",
        batch_size=config.batch_size,
        max_len=config.max_len,
        limit=config.train_limit_data
    )

    valid_loader = create_dataloader(
        "validation",
        batch_size=config.batch_size,
        max_len=config.max_len,
        limit=config.val_limit_data
    )
    print(f"✓ DataLoaders ready. (Train: {len(train_loader)} batches)")

    model, device = build_model(config)

    criterion = nn.CrossEntropyLoss(
        ignore_index=trg_pad_idx,
        label_smoothing=0.1
    )

    if config.use_noam:
        base_opt = Adam(model.parameters(), lr=0)
        optimizer = NoamOpt(
            d_model=config.d_model,
            warmup_step=config.warmup_steps,
            optimizer=base_opt
        )
    else:
        optimizer = Adam(model.parameters(), lr=config.fixed_lr)


    for epoch in range(config.epochs):

        attention_filename = f"attention_{config.start_Temp}_{config.end_Temp}_{config.schedule}_{config.seed}_{epoch+1}.pkl"
        attention_path_file = f"saved_attention_data/{attention_filename}"


        print(f"Epoch [{epoch+1}/{config.epochs}] - Training...")
        train_loss, train_ppl, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion,
            config, device, global_step
        )

        print(f"Epoch [{epoch+1}/{config.epochs}] - Running Evaluation...")
        val_loss, val_ppl, bleu = evaluate(
            model, valid_loader, criterion,
            config, device, global_step,dump_attention=config.dump_attention,
            save_path_file=attention_path_file if config.dump_attention else None
        )

        print(f"Epoch {epoch+1}")
        print(f"Train PPL: {train_ppl:.3f} | "
              f"Val PPL: {val_ppl:.3f} | BLEU: {bleu:.3f}")

        torch.save(
            model.state_dict(),
            f"saved_model/model_epoch{epoch+1}.pt"
        )