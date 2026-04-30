import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import BabiDataset
from model import MemN2N, prepare_data

def train_task(task_id, epochs=100, batch_size=32, verbose=True,
               use_ls=True, use_pe=False, use_rn=False, hops=3,
               ls_lr=0.005, post_ls_lr=0.01,
               ls_patience=5, val_frac=0.1, seed=0,
               tying='adjacent', use_relu=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_task{task_id}.pth")

    dataset = BabiDataset(download=True)
    stories_train, questions_train, vocab = dataset.load_task(task_id, train=True)
    stories_test, questions_test, _ = dataset.load_task(task_id, train=False)

    X_train_story, X_train_query, Y_train = prepare_data(questions_train, stories_train, vocab)
    X_test_story, X_test_query, Y_test = prepare_data(questions_test, stories_test, vocab)

    g = torch.Generator().manual_seed(seed)
    n = len(Y_train)
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n, generator=g)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xs_tr, Xq_tr, Y_tr = X_train_story[tr_idx], X_train_query[tr_idx], Y_train[tr_idx]
    Xs_val, Xq_val, Y_val = X_train_story[val_idx], X_train_query[val_idx], Y_train[val_idx]

    vocab_size = len(vocab)
    model = MemN2N(vocab_size=vocab_size, embed_size=20, max_story_len=50, hops=hops,
                   use_pe=use_pe, use_rn=use_rn, tying=tying, use_relu=use_relu)

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    ls_phase = use_ls
    model.use_softmax = not ls_phase
    init_lr = ls_lr if ls_phase else post_ls_lr
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    best_val_acc = -1.0
    final_test_acc = 0.0
    best_val_loss = float("inf")
    bad_epochs = 0

    if verbose:
        phase = "LS (linear)" if ls_phase else "standard"
        print(f"\nTraining MemN2N on Task {task_id} (vocab={vocab_size}) — phase: {phase}")

    num_samples = len(Y_tr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i in range(0, num_samples, batch_size):
            batch_story = Xs_tr[i:i+batch_size]
            batch_query = Xq_tr[i:i+batch_size]
            batch_y = Y_tr[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(batch_story, batch_query)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xs_val, Xq_val)
            val_loss = criterion(val_logits, Y_val).item() / max(1, len(Y_val))
            val_preds = torch.argmax(val_logits, dim=1)
            current_val_acc = (val_preds == Y_val).sum().item() / max(1, len(Y_val))

            test_logits = model(X_test_story, X_test_query)
            test_preds = torch.argmax(test_logits, dim=1)
            current_test_acc = (test_preds == Y_test).sum().item() / len(Y_test)

        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            final_test_acc = current_test_acc
            checkpoint = {'state_dict': model.state_dict(), 'vocab': vocab,
                          'use_pe': use_pe, 'use_rn': use_rn, 'hops': hops,
                          'tying': tying, 'use_relu': use_relu}
            torch.save(checkpoint, save_path)
            saved_marker = " --> Model Saved!"
        else:
            saved_marker = ""

        if ls_phase:
            if val_loss + 1e-4 < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= ls_patience:
                ls_phase = False
                model.use_softmax = True
                optimizer = optim.SGD(model.parameters(), lr=post_ls_lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
                best_val_loss = float("inf")
                bad_epochs = 0
                if verbose:
                    print(f"Epoch {epoch+1:03d}: val loss plateaued — re-inserting softmaxes")

        if verbose and (saved_marker or (epoch + 1) % 10 == 0):
            tag = "LS" if ls_phase else "SM"
            print(f"[{tag}] Epoch {epoch+1:03d}/{epochs} | "
                  f"TrainLoss: {total_loss/num_samples:.4f} | "
                  f"ValLoss: {val_loss:.4f} | "
                  f"ValAcc: {current_val_acc:.4f} | "
                  f"TestAcc: {current_test_acc:.4f}{saved_marker}")

    return best_val_acc, final_test_acc

def train_joint(epochs=60, batch_size=32, verbose=True,
                use_ls=True, use_pe=False, use_rn=False, hops=3,
                ls_lr=0.005, post_ls_lr=0.01,
                ls_patience=5, val_frac=0.1, seed=0,
                embed_size=50, anneal_step=15,
                tying='adjacent', use_relu=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model_joint.pth")

    # Load Data
    dataset = BabiDataset(download=True)
    stories_train, questions_train, vocab = dataset.load_all_tasks_joint(train=True)
    stories_test, questions_test, _ = dataset.load_all_tasks_joint(train=False)

    all_stories = stories_train + stories_test
    all_questions = questions_train + questions_test
    max_sen_len = max(len(BabiDataset.tokenize(s)) for st in all_stories for s in st["sentences"])
    max_q_len = max(len(BabiDataset.tokenize(q["question"])) for q in all_questions)

    # Prepare Tensors
    X_train_story, X_train_query, Y_train = prepare_data(
        questions_train, stories_train, vocab,
        max_sen_len=max_sen_len, max_q_len=max_q_len)
    X_test_story, X_test_query, Y_test = prepare_data(
        questions_test, stories_test, vocab,
        max_sen_len=max_sen_len, max_q_len=max_q_len)

    task_ids_test = torch.tensor([q["task_id"] for q in questions_test])

    #  Validation Split
    g = torch.Generator().manual_seed(seed)
    n = len(Y_train)
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n, generator=g)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    
    Xs_tr, Xq_tr, Y_tr = X_train_story[tr_idx], X_train_query[tr_idx], Y_train[tr_idx]
    Xs_val, Xq_val, Y_val = X_train_story[val_idx], X_train_query[val_idx], Y_train[val_idx]

    #  Model Setup
    model = MemN2N(vocab_size=len(vocab), embed_size=embed_size,
                   max_story_len=50, hops=hops,
                   use_pe=use_pe, use_rn=use_rn,
                   tying=tying, use_relu=use_relu)

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    ls_phase = use_ls
    model.use_softmax = not ls_phase
    optimizer = optim.SGD(model.parameters(), lr=ls_lr if ls_phase else post_ls_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=anneal_step, gamma=0.5)

    best_val_acc = -1.0
    best_per_task = None
    best_val_loss = float("inf")
    bad_epochs = 0

    #  Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_perm = torch.randperm(len(Y_tr), generator=g)

        for i in range(0, len(Y_tr), batch_size):
            idx = epoch_perm[i:i+batch_size]
            optimizer.zero_grad()
            logits = model(Xs_tr[idx], Xq_tr[idx])
            loss = criterion(logits, Y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()


        model.eval()
        with torch.no_grad():
            val_losses = []
            val_correct = 0
            for i in range(0, len(Y_val), 64):
                v_logits = model(Xs_val[i:i+64], Xq_val[i:i+64])
                val_losses.append(criterion(v_logits, Y_val[i:i+64]))
                val_correct += (torch.argmax(v_logits, dim=1) == Y_val[i:i+64]).sum().item()
            val_loss = sum(val_losses).item() / len(Y_val)
            current_val_acc = val_correct / max(1, len(Y_val))

            all_preds = []
            for i in range(0, len(Y_test), 64):
                t_logits = model(X_test_story[i:i+64], X_test_query[i:i+64])
                all_preds.append(torch.argmax(t_logits, dim=1))
            test_preds = torch.cat(all_preds)

            per_task = {}
            for tid in range(1, 21):
                mask = task_ids_test == tid
                per_task[tid] = (test_preds[mask] == Y_test[mask]).float().mean().item()

            mean_acc = sum(per_task.values()) / 20

        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_per_task = dict(per_task)
            torch.save({'state_dict': model.state_dict(), 'vocab': vocab,
                        'hops': hops, 'embed_size': embed_size,
                        'use_pe': use_pe, 'use_rn': use_rn,
                        'tying': tying, 'use_relu': use_relu}, save_path)

        # Linear Start plateau detection, re-insert softmaxes when val loss stops improving
        if ls_phase:
            if val_loss + 1e-4 < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= ls_patience:
                ls_phase = False
                model.use_softmax = True
                optimizer = optim.SGD(model.parameters(), lr=post_ls_lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=anneal_step, gamma=0.5)
                best_val_loss = float("inf")
                bad_epochs = 0
                if verbose:
                    print(f"Epoch {epoch+1:03d}: val loss plateaued — re-inserting softmaxes")

        if verbose and (epoch + 1) % 5 == 0:
            tag = "LS" if ls_phase else "SM"
            print(f"[{tag}] Epoch {epoch+1:03d} | TrainLoss: {total_loss/len(Y_tr):.4f} | "
                  f"ValAcc: {current_val_acc:.4f} | MeanAcc: {mean_acc:.4f}")

    return best_val_acc, best_per_task

if __name__ == "__main__":
    task = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    train_task(task, verbose=True)
