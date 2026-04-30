import os
import torch
import shutil
from data.dataloader import BabiDataset
from .train import train_multiword
from .eval import evaluate_multiword

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