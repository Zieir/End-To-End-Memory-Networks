import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import BabiDataset
from model import MemN2N_MultiWord, prepare_data_multi

def train_multiword(task_id, epochs=100, batch_size=32, max_ans_len=5, seed=0):
    # Initialisation de la graine pour le Best-of-N
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir = "./models_multi"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_task{task_id}_multi.pth")

    dataset = BabiDataset(download=True)
    stories_train, questions_train, vocab = dataset.load_task(task_id, train=True)
    stories_test, questions_test, _ = dataset.load_task(task_id, train=False)

    X_train_story, X_train_query, Y_train = prepare_data_multi(questions_train, stories_train, vocab, max_ans_len=max_ans_len)
    X_test_story, X_test_query, Y_test = prepare_data_multi(questions_test, stories_test, vocab, max_ans_len=max_ans_len)

    model = MemN2N_MultiWord(vocab_size=len(vocab), max_ans_len=max_ans_len)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i in range(0, len(Y_train), batch_size):
            optimizer.zero_grad()
            logits = model(X_train_story[i:i+batch_size], X_train_query[i:i+batch_size])
            
            loss = criterion(logits.view(-1, len(vocab)), Y_train[i:i+batch_size].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_story, X_test_query).argmax(dim=2)
            correct = torch.all(test_preds == Y_test, dim=1).sum().item()
            acc = correct / len(Y_test)

        if acc >= best_acc:
            best_acc = acc
            torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'max_ans_len': max_ans_len}, save_path)
            
        # Optionnel: décommenter pour voir les logs pendant l'entraînement
        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch {epoch+1} | Loss: {total_loss/len(Y_train):.4f} | Accuracy: {acc:.4f}")

    return best_acc # INDISPENSABLE pour run_all_multi.py !

if __name__ == "__main__":
    train_multiword(task_id=8)