import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.preprocess import preprocess_data, tokenize
from src.dataset import SeqDataset
from src.models.seq2seq import Seq2Seq
from src.utils import set_seed
from src.inference import translate_sentence

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10, device='cuda', early_stopping_patience=3):
    model.to(device)
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as tepoch:
            for src, tgt in tepoch:
                src, tgt = src.to(device), tgt.to(device)
                inp = tgt[:, :-1]
                target = tgt[:, 1:]
                optimizer.zero_grad()
                output = model(src, inp)
                loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tepoch.set_postfix({'Train Loss': f"{loss.item():.3f}"})
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

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
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break

    return train_losses, val_losses, best_val_loss

def encode_sentence(sentence, vocab, max_len=50):
    """
    Tokenize and numericalize a single user-input sentence.
    """
    tokens = tokenize(sentence)

    indexed = [vocab.get('<sos>')] + [vocab.get(w, vocab['<unk>']) for w in tokens] + [vocab.get('<eos>')]

    if len(indexed) < max_len:
        indexed += [vocab['<pad>']] * (max_len - len(indexed))
    else:
        indexed = indexed[:max_len]
    return torch.tensor(indexed, dtype=torch.long).unsqueeze(0)

if __name__ == '__main__':
    set_seed(42)

    csv_path = 'data/sentences.csv'
    (src_train, tgt_train), (src_val, tgt_val), src_vocab, tgt_vocab = preprocess_data(csv_path, src_col='eng', tgt_col='darija_ar', max_len=50)
    train_dataset = SeqDataset(src_train, tgt_train)
    val_dataset = SeqDataset(src_val, tgt_val)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    embed_size = 256
    hidden_size = 512
    learning_rate = 0.001
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout_rate = 0.3
    early_stopping_patience = 2

    pad_idx = tgt_vocab['<pad>']
    model_vanilla = Seq2Seq(len(src_vocab), len(tgt_vocab), embed_size, hidden_size, lstm_cell_type='vanilla', dropout_rate=dropout_rate)
    optimizer = optim.Adam(model_vanilla.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_losses, val_losses, best_val_loss = train_model(
        model_vanilla, train_loader, val_loader, optimizer, criterion,
        epochs=epochs, device=device, early_stopping_patience=early_stopping_patience
    )


    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


    while True:
        user_input = input("Enter an English sentence to translate (or type 'quit' to exit):\n> ")
        if user_input.lower().strip() == 'quit':
            break

        model_vanilla.eval()

        src_tensor = encode_sentence(user_input, src_vocab, max_len=50).to(device)
        predicted_translation = translate_sentence(model_vanilla, src_tensor, tgt_vocab, device=device)

        print("Predicted Darija Arabic:", predicted_translation)
        print("-"*50)
