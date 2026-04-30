import os
import torch
from data.dataloader import BabiDataset
from train import train_task
from eval import evaluate_task

def run_best_of_n(n_runs=10, use_ls=True, use_pe=True, use_rn=False, hops=3, epochs=100):
    dataset = BabiDataset(download=False)
    results = {}
    
    # Directory for storing models
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting 'Best of {n_runs}' training across all 20 tasks...")

    for task_id in range(1, 21):
        task_desc = dataset.TASK_DESCRIPTIONS[task_id]
        print(f"\nPROCESSING TASK {task_id}: {task_desc}\n")
        
        best_acc_for_task = -1.0
        
        for run in range(n_runs):
            seed = run  # Use run index as seed for reproducibility
            print(f"\n--- Task {task_id} | Run {run+1}/{n_runs} (Seed: {seed}) ---")
            
            # Train the model for this run
            # train_task saves the model to ./models/best_model_task{task_id}.pth 
            # internally if it beats its own internal validation during that run
            current_run_acc = train_task(
                task_id, 
                epochs=epochs, 
                verbose=False,
                use_ls=use_ls, 
                use_pe=use_pe, 
                use_rn=use_rn,
                hops=hops, 
                seed=seed
            )
            
            # Evaluate the model saved by this run
            # We compare it against the absolute best found across all previous 10-run attempts
            if current_run_acc > best_acc_for_task:
                best_acc_for_task = current_run_acc
                # Rename/keep this as the absolute gold model for this task
                gold_path = os.path.join(save_dir, f"gold_model_task{task_id}.pth")
                temp_path = os.path.join(save_dir, f"best_model_task{task_id}.pth")
                
                # Check if the file exists before attempting to copy/move
                if os.path.exists(temp_path):
                    import shutil
                    shutil.copyfile(temp_path, gold_path)
                
                print(f"New Absolute Best for Task {task_id}: {best_acc_for_task:.4f}")
            else:
                print(f"Run {run+1} result ({current_run_acc:.4f}) did not beat best ({best_acc_for_task:.4f})")

        results[task_id] = best_acc_for_task

    # Final Summary
    print(f"\nFINAL BEST-OF-{n_runs} SUMMARY\n")
    for tid, acc in results.items():
        print(f"Task {tid:02d}: {acc:.4f} accuracy")

if __name__ == "__main__":
    run_best_of_n(
        n_runs=10, 
        use_ls=True, 
        use_pe=True, 
        hops=3, 
        epochs=100
    )