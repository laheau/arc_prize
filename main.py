from torch.utils.data import DataLoader, Dataset

import torch
from src.gptcode import MaskedAutoencoderTwoView

# Dummy dataset example
class DummyTwoViewDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=32, in_chans=12):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.in_chans = in_chans

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Replace this with real data (two views of same object)
        x = torch.randn(2, self.in_chans, self.img_size, self.img_size)
        return x  # (2,12,32,32)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = DummyTwoViewDataset(num_samples=1000, img_size=32, in_chans=12)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MaskedAutoencoderTwoView(
        img_size=32,
        in_chans=12,
        embed_dim=256,
        depth=6,
        num_heads=8,
        patch_size=4,
        mask_ratio=0.6,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            # batch: (B, 2, 12, 32, 32)
            x = batch.to(device)
            loss, pred, mask = model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")


if __name__ == "__main__":
    train()
