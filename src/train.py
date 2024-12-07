

import torch

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            inp = tgt[:, :-1]
            target = tgt[:, 1:]
            optimizer.zero_grad()
            output = model(src, inp)
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                inp = tgt[:, :-1]
                target = tgt[:, 1:]
                output = model(src, inp)
                loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f}")
