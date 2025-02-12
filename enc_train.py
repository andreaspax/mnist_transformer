import torch
import wandb

torch.manual_seed(2)


batch_size = 512
embedding_dim = 64
epochs = 10
initial_lr = 0.001


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

wandb.init(
    project="mlx6-mnist-transformer",
    config={
        "learning_rate": initial_lr,
        "epochs": epochs,
    },
)

model = 

print("param count:", sum(p.numel() for p in model.parameters()))

model.to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")
    prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for center, context in prgs:
        center, context = center.to(device), context.to(device)

        cbow_rand = torch.randint(0, vocab_size, (center.size(0),)).to(device)
        skipgram_rand = torch.randint(0, vocab_size, (context.size(0), 2)).to(
            device
        )

        if arch == "cbow":
            rand = cbow_rand
        else:
            rand = skipgram_rand

        optimiser.zero_grad()
        loss = model(center, context, rand)

        loss.backward()
        optimiser.step()

        wandb.log({"loss": loss.item()})

torch.save(model.state_dict(), os.path.join(script_dir, "weights.pt"))
print("Uploading...")
artifact = wandb.Artifact("model-weights", type="model")
artifact.add_file("./weights.pt")
wandb.log_artifact(artifact)
print("Done!")
wandb.finish()


train_embeddings("sources/normalised_corpus_1.txt")
train_embeddings("sources/normalised_corpus_2.txt")