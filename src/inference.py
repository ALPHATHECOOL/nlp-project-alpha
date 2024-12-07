
import torch

def translate_sentence(model, src_sentence, tgt_vocab, max_len=50, device='cuda'):
    model.eval()
    with torch.no_grad():
        h, c = model.encoder(src_sentence)
        sos_idx = tgt_vocab['<sos>']
        eos_idx = tgt_vocab['<eos>']
        pred = [sos_idx]
        for _ in range(max_len):
            inp = torch.tensor([pred[-1]], device=device).unsqueeze(0)
            x_t = model.decoder.embedding(inp)
            h, c = model.decoder.lstm_cell(x_t.squeeze(1), h, c)
            logits = model.decoder.fc(h)
            next_token = logits.argmax(-1).item()
            pred.append(next_token)
            if next_token == eos_idx:
                break
    inv_tgt_vocab = {i: w for w, i in tgt_vocab.items()}

    predicted_sentence = ' '.join(inv_tgt_vocab.get(w, '<unk>') for w in pred[1:-1])
    return predicted_sentence
