import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import BabiDataset
from model import MemN2N, prepare_data

def train_task(task_id, epochs=100, batch_size=32, verbose=True):
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_task{task_id}.pth")

    dataset = BabiDataset(download=True)
    stories_train, questions_train, vocab = dataset.load_task(task_id, train=True)
    stories_test, questions_test, _ = dataset.load_task(task_id, train=False)
    
    X_train_story, X_train_query, Y_train = prepare_data(questions_train, stories_train, vocab)
    X_test_story, X_test_query, Y_test = prepare_data(questions_test, stories_test, vocab)
    
    vocab_size = len(vocab)
    model = MemN2N(vocab_size=vocab_size, embed_size=20, max_story_len=50, hops=3)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum') 
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    num_samples = len(Y_train)
    best_test_acc = 0.0 
    
    if verbose:
        print(f"\nTraining MemN2N on Task {task_id} with vocabulary size {vocab_size}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, num_samples, batch_size):
            batch_story = X_train_story[i:i+batch_size]
            batch_query = X_train_query[i:i+batch_size]
            batch_y = Y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            logits = model(batch_story, batch_query)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0) 
            optimizer.step()
            
            total_loss += loss.item() 
            
        scheduler.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_story, X_test_query)
            test_preds = torch.argmax(test_logits, dim=1)
            current_test_acc = (test_preds == Y_test).sum().item() / len(Y_test)
            
        if current_test_acc > best_test_acc:
            best_test_acc = current_test_acc
            checkpoint = {
                'state_dict': model.state_dict(),
                'vocab': vocab
            }
            torch.save(checkpoint, save_path)
            if verbose:
                print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/num_samples:.4f} | Test Acc: {current_test_acc:.4f} --> Model Saved!")
        elif verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/num_samples:.4f} | Test Acc: {current_test_acc:.4f}")
            
    return best_test_acc

if __name__ == "__main__":
    task = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    train_task(task, verbose=True)