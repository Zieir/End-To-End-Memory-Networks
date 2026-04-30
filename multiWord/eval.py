import torch
from data.dataloader import BabiDataset
from model import prepare_data_multi, MemN2N_MultiWord

def evaluate_multiword(task_id):
    checkpoint = torch.load(f"./models_multi/best_model_task{task_id}_multi.pth")
    vocab = checkpoint['vocab']
    max_ans_len = checkpoint['max_ans_len']
    
    model = MemN2N_MultiWord(vocab_size=len(vocab), max_ans_len=max_ans_len)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = BabiDataset(download=False)
    stories_ts, questions_ts, _ = dataset.load_task(task_id, train=False)
    Xs, Xq, Y = prepare_data_multi(questions_ts, stories_ts, vocab, max_ans_len=max_ans_len)
    
    with torch.no_grad():
        preds = model(Xs, Xq).argmax(dim=2)
        acc = torch.all(preds == Y, dim=1).float().mean().item()[cite: 14]
        
    print(f"Accuracy Task {task_id}: {acc:.4f}")
    
    # Affichage d'un exemple[cite: 14]
    idx2word = {i: w for w, i in vocab.items()}
    pred_sent = [idx2word[i.item()] for i in preds[0] if i > 0]
    true_sent = [idx2word[i.item()] for i in Y[0] if i > 0]
    print(f"Exemple - Pred: {' '.join(pred_sent)} | True: {' '.join(true_sent)}")

if __name__ == "__main__":
    evaluate_multiword(8)