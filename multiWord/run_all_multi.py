import os
import torch
import shutil
from data.dataloader import BabiDataset
from train_3 import train_multiword
from eval_3 import evaluate_multiword

def run_pipeline_multi(n_runs=10, epochs=100):
    dataset = BabiDataset(download=False)
    
    # Storage directories
    tmp_dir = "./models_multi"
    gold_dir = "./models_multi_word"
    os.makedirs(gold_dir, exist_ok=True)

    print(f"Starting 'Best of {n_runs}' training across all 20 tasks with multi-word answers...\n")

    best_scores = {}

    for task_id in range(1, 21):
        task_desc = dataset.TASK_DESCRIPTIONS[task_id]
        print(f"{'='*60}")
        print(f"TASK {task_id:02d} : {task_desc}")
        print(f"{'='*60}")
        
        best_acc_found = -1.0
        
        for run in range(n_runs):
            seed = run
            print(f"\n--- Run {run+1}/{n_runs} (Seed: {seed}) ---")
            
            # Training utilizes Position Encoding (PE) and Linear Start (LS) internally[cite: 15, 16]
            current_acc = train_multiword(
                task_id=task_id, 
                epochs=epochs, 
                seed=seed,
                max_ans_len=5
            )[cite: 16]
            
            # Update absolute best for this specific task
            if current_acc > best_acc_found:
                best_acc_found = current_acc
                
                # Copy the temporary best of this run to the final gold directory[cite: 16]
                src_path = f"./models_multi/best_model_task{task_id}_multi.pth"
                dst_path = os.path.join(gold_dir, f"gold_model_task{task_id}_multi.pth")
                
                if os.path.exists(src_path):
                    shutil.copyfile(src_path, dst_path)
                
                print(f"New Absolute Best for Task {task_id} : {best_acc_found:.4f} accuracy")
            else:
                print(f"Run {run+1} result ({current_acc:.4f}) did not beat (Best: {best_acc_found:.4f})")

        best_scores[task_id] = {
            'description': task_desc,
            'accuracy': best_acc_found
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