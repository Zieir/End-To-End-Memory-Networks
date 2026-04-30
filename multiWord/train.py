import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import BabiDataset
from .model import MemN2N_MultiWord, prepare_data_multi

def train_multiword(task_id, epochs=100, batch_size=32, max_ans_len=5,
                    val_frac=0.1, seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir = "./models_multi"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_task{task_id}_multi.pth")

    dataset = BabiDataset(download=True)
    stories_train, questions_train, vocab = dataset.load_task(task_id, train=True)
    stories_test, questions_test, _ = dataset.load_task(task_id, train=False)

    X_train_story, X_train_query, Y_train = prepare_data_multi(
        questions_train, stories_train, vocab, max_ans_len=max_ans_len)
    X_test_story, X_test_query, Y_test = prepare_data_multi(
        questions_test, stories_test, vocab, max_ans_len=max_ans_len)

    g = torch.Generator().manual_seed(seed)
    n = len(Y_train)
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n, generator=g)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xs_tr, Xq_tr, Y_tr = X_train_story[tr_idx], X_train_query[tr_idx], Y_train[tr_idx]
    Xs_val, Xq_val, Y_val = X_train_story[val_idx], X_train_query[val_idx], Y_train[val_idx]

    model = MemN2N_MultiWord(vocab_size=len(vocab), max_ans_len=max_ans_len)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Use mean reduction to get proper loss per example
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

    best_val_acc = -1.0
    final_test_acc = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle training data
        perm_train = torch.randperm(len(Y_tr), generator=g)
        
        for i in range(0, len(Y_tr), batch_size):
            idx = perm_train[i:i+batch_size]
            optimizer.zero_grad()
            
            logits = model(Xs_tr[idx], Xq_tr[idx])  # [batch, max_ans_len, vocab_size]
            # Reshape for loss computation
            loss = criterion(logits.view(-1, len(vocab)), Y_tr[idx].view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(Xs_val, Xq_val).argmax(dim=2)
            # Check if entire sequence matches (all tokens correct)
            val_acc = torch.all(val_preds == Y_val, dim=1).float().mean().item()

            test_preds = model(X_test_story, X_test_query).argmax(dim=2)
            test_acc = torch.all(test_preds == Y_test, dim=1).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            patience_counter = 0
            torch.save({'state_dict': model.state_dict(), 'vocab': vocab,
                        'max_ans_len': max_ans_len}, save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"ValAcc: {val_acc:.4f} | TestAcc: {test_acc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_val_acc, final_test_acc

if __name__ == "__main__":
    train_multiword(task_id=8)