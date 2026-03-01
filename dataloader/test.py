from dataloader.dataloader import create_dataloader

train_loader = create_dataloader("train", batch_size=64, max_len=128)

batch = next(iter(train_loader))

print("Keys:", batch.keys())
print("src shape:", batch["src"].shape)
print("tgt shape:", batch["tgt"].shape)

print("src[0]:", batch["src"][0])
print("tgt[0]:", batch["tgt"][0])