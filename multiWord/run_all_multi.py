""""
import os
import sys

sysPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if sysPath not in sys.path:
    sys.path.append(sysPath)
import torch
import shutil
from data.dataloader import BabiDataset
from multiWord.train import train_multiword
from multiWord.eval import evaluate_multiword

def run_pipeline_multi(n_runs=10, epochs=100):
    dataset = BabiDataset(download=False)

    gold_dir = "./models_multi_word"
    os.makedirs(gold_dir, exist_ok=True)

    print(f"Starting 'Best of {n_runs}' training across all 20 tasks with multi-word answers...\n")

    best_scores = {}

    for task_id in range(1, 21):
        task_desc = dataset.TASK_DESCRIPTIONS[task_id]
        print(f"{'='*60}")
        print(f"TASK {task_id:02d} : {task_desc}")
        print(f"{'='*60}")

        best_val_found = -1.0
        best_test_for_task = 0.0

        for run in range(n_runs):
            seed = run
            print(f"\n--- Run {run+1}/{n_runs} (Seed: {seed}) ---")

            run_val_acc, run_test_acc = train_multiword(
                task_id=task_id,
                epochs=epochs,
                seed=seed,
                max_ans_len=5
            )

            # Best-of-N selection on validation accuracy (no test-set leakage)
            if run_val_acc > best_val_found:
                best_val_found = run_val_acc
                best_test_for_task = run_test_acc

                src_path = f"./models_multi/best_model_task{task_id}_multi.pth"
                dst_path = os.path.join(gold_dir, f"gold_model_task{task_id}_multi.pth")

                if os.path.exists(src_path):
                    shutil.copyfile(src_path, dst_path)

                print(f"New Best for Task {task_id}: val={best_val_found:.4f}, "
                      f"test={best_test_for_task:.4f}")
            else:
                print(f"Run {run+1} val={run_val_acc:.4f} did not beat best val "
                      f"({best_val_found:.4f})")

        best_scores[task_id] = {
            'description': task_desc,
            'accuracy': best_test_for_task
        }

    # Final Results Table Summary
    print(f"{'Task':<4} | {'Description':<30} | {'Best Acc':<10} | {'Error %':<10}")
    
    total_error = 0
    for tid in range(1, 21):
        desc = best_scores[tid]['description']
        acc = best_scores[tid]['accuracy']
        err = (1 - acc) * 100
        total_error += err
        print(f"{tid:<4} | {desc:<30} | {acc:<10.4f} | {err:>7.1f}%")
    
    print(f"Mean Error Rate: {total_error / 20:>49.1f}%")

if __name__ == "__main__":
    run_pipeline_multi(n_runs=10, epochs=100)
    """
import os
import sys 

sysPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if sysPath not in sys.path:
    sys.path.append(sysPath)

import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from data.dataloader import BabiDataset
from multiWord.model import MemN2N_MultiWord, prepare_data_multi

def train_joint_multiword(epochs=60, batch_size=32, n_runs=5, val_frac=0.1, embed_size=50, hops=3):
    """
    Entraîne un modèle génératif unique sur les 20 tâches simultanément.
    Répète l'opération n_runs fois pour conserver la meilleure initialisation.
    """
    save_dir = "./models_multi_word"
    os.makedirs(save_dir, exist_ok=True)
    gold_path = os.path.join(save_dir, "gold_model_joint_multi.pth")

    print("Chargement du corpus Joint (20 tâches)...")
    dataset = BabiDataset(download=True)
    stories_train, questions_train, vocab = dataset.load_all_tasks_joint(train=True)
    stories_test, questions_test, _ = dataset.load_all_tasks_joint(train=False)

    vocab_size = len(vocab)
    
    # 1. Calcul dynamique CORRIGÉ de la longueur max de réponse (séparation par virgule)
    all_questions = questions_train + questions_test
    max_ans_len = max(len([t for t in q["answer"].split(",") if t.strip()]) for q in all_questions)
    
    # 2. Fixer les dimensions globales pour éviter les crashs Train vs Test
    all_stories = stories_train + stories_test
    max_sen_len = max(len(BabiDataset.tokenize(s)) for st in all_stories for s in st["sentences"])
    max_q_len = max(len(BabiDataset.tokenize(q["question"])) for q in all_questions)

    print(f"Vocabulaire global : {vocab_size} mots | Longueur max de réponse : {max_ans_len} mots")

    # Utilisation des paramètres globaux
    X_train_story, X_train_query, Y_train = prepare_data_multi(
        questions_train, stories_train, vocab, 
        max_sen_len=max_sen_len, max_q_len=max_q_len, max_ans_len=max_ans_len)
    
    X_test_story, X_test_query, Y_test = prepare_data_multi(
        questions_test, stories_test, vocab, 
        max_sen_len=max_sen_len, max_q_len=max_q_len, max_ans_len=max_ans_len)


    vocab_size = max(vocab.values()) + 1 
    print(f"Vocabulaire global final après parsing : {vocab_size} mots")
    
    task_ids_test = torch.tensor([q["task_id"] for q in questions_test])
    num_samples = len(Y_train)

    best_global_mean_acc = -1.0
    best_global_per_task = None

    print(f"\nDémarrage de l'entraînement Joint Multi-Word ({n_runs} tentatives)")

    for run in range(n_runs):
        seed = run
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        print(f"\n{'-'*50}\nTentative Joint {run+1}/{n_runs} (Seed: {seed})\n{'-'*50}")

        # Séparation Train/Val
        g = torch.Generator().manual_seed(seed)
        n_val = max(1, int(num_samples * val_frac))
        perm = torch.randperm(num_samples, generator=g)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        
        Xs_tr, Xq_tr, Y_tr = X_train_story[tr_idx], X_train_query[tr_idx], Y_train[tr_idx]

        model = MemN2N_MultiWord(vocab_size=vocab_size, embed_size=embed_size, 
                                 hops=hops, max_ans_len=max_ans_len)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        best_run_mean_acc = 0.0

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            epoch_perm = torch.randperm(len(Y_tr), generator=g)
            
            for i in range(0, len(Y_tr), batch_size):
                idx = epoch_perm[i:i+batch_size]
                optimizer.zero_grad()
                
                logits = model(Xs_tr[idx], Xq_tr[idx])
                loss = criterion(logits.view(-1, vocab_size), Y_tr[idx].view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0)
                optimizer.step()
                total_loss += loss.item()

            # --- ÉVALUATION SÉCURISÉE EN MÉMOIRE ---
            model.eval()
            with torch.no_grad():
                all_preds = []
                for i in range(0, len(Y_test), 64):
                    t_logits = model(X_test_story[i:i+64], X_test_query[i:i+64])
                    all_preds.append(torch.argmax(t_logits, dim=2)) 
                
                test_preds = torch.cat(all_preds, dim=0)

                per_task = {}
                for tid in range(1, 21):
                    mask = task_ids_test == tid
                    if mask.any():
                        task_preds = test_preds[mask]
                        task_targets = Y_test[mask]
                        
                        # Compare only the non-padding tokens[cite: 16, 17]
                        # A prediction is correct if it matches the target wherever the target is not 0
                        valid_mask = task_targets != 0 
                        
                        # For an answer to be perfectly correct, ALL valid tokens must match
                        matches = (task_preds == task_targets) | ~valid_mask
                        correct = torch.all(matches, dim=1).sum().item()
                        
                        per_task[tid] = correct / mask.sum().item()
                
                mean_acc = sum(per_task.values()) / 20
                
            if mean_acc > best_run_mean_acc:
                best_run_mean_acc = mean_acc
                if mean_acc > best_global_mean_acc:
                    best_global_mean_acc = mean_acc
                    best_global_per_task = dict(per_task)
                    torch.save({
                        'state_dict': model.state_dict(),
                        'vocab': vocab,
                        'max_ans_len': max_ans_len,
                        'embed_size': embed_size,
                        'hops': hops
                    }, gold_path)
                    marker = " 🌟 [NOUVEAU RECORD GLOBAL]"
                else:
                    marker = " *"
            else:
                marker = ""

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(Y_tr):.4f} | MeanAcc: {mean_acc:.4f}{marker}")

    # Résumé Final
    print(f"\n\n{'='*60}")
    print(f"RÉSUMÉ FINAL DU MEILLEUR MODÈLE JOINT (Accuracy Moyenne: {best_global_mean_acc:.4f})")
    print(f"{'='*60}")
    print(f"{'Task':<4} | {'Description':<30} | {'Test Acc':<10} | {'Error %':<10}")
    
    total_error = 0
    for tid in range(1, 21):
        desc = dataset.TASK_DESCRIPTIONS[tid]
        acc = best_global_per_task[tid]
        err = (1 - acc) * 100
        total_error += err
        print(f"{tid:<4} | {desc:<30} | {acc:<10.4f} | {err:>7.1f}%")
    print(f"{'-'*60}")
    print(f"Mean Error Rate: {total_error / 20:>34.1f}%")

if __name__ == "__main__":
    train_joint_multiword(epochs=60, batch_size=32, n_runs=5)