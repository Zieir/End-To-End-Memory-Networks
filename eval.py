import os
import sys
import torch
from data.dataloader import BabiDataset
from model import MemN2N, prepare_data

def evaluate_task(task_id, verbose=True):
    load_path = f"./models/best_model_task{task_id}.pth"
    
    if not os.path.exists(load_path):
        if verbose: print(f"Error: Could not find model at {load_path}.")
        return 0.0

    checkpoint = torch.load(load_path)
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)
    use_pe = checkpoint.get('use_pe', False)

    model = MemN2N(vocab_size=vocab_size, embed_size=20, max_story_len=50, hops=3,
                   use_pe=use_pe)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval() 

    dataset = BabiDataset(download=False)
    stories_test, questions_test, _ = dataset.load_task(task_id, train=False)
    X_test_story, X_test_query, Y_test = prepare_data(questions_test, stories_test, vocab)
    
    with torch.no_grad():
        test_logits = model(X_test_story, X_test_query)
        test_preds = torch.argmax(test_logits, dim=1)
        correct_test = (test_preds == Y_test).sum().item()
        test_acc = correct_test / len(Y_test)
        
    if verbose:
        print(f"Final Evaluation on Task {task_id}")
        print(f"Accuracy:   {test_acc:.4f}")
        print(f"Error Rate: {(1 - test_acc) * 100:.1f}%")
        
    return test_acc

if __name__ == "__main__":
    task = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    evaluate_task(task, verbose=True)