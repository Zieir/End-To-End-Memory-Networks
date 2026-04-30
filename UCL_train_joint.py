import os
import torch
import shutil
import argparse
from data.dataloader import BabiDataset
from train_miniBatch import train_task, train_joint
from eval import evaluate_task

def run_best_of_n(n_runs=10, use_ls=True, use_pe=True, use_rn=False, hops=3, epochs=100):
    """Exécute l'entraînement indépendant pour chaque tâche avec sélection du meilleur run[cite: 18]."""
    dataset = BabiDataset(download=False)
    results = {}
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting 'Best of {n_runs}' training across all 20 tasks...\n")

    for task_id in range(1, 21):
        task_desc = dataset.TASK_DESCRIPTIONS[task_id]
        print(f"PROCESSING TASK {task_id}: {task_desc}")
        
        best_acc_for_task = -1.0
        
        for run in range(n_runs):
            seed = run 
            print(f"  -> Run {run+1}/{n_runs} (Seed: {seed})")
            
            # Entraînement individuel
            current_run_acc = train_task(
                task_id, epochs=epochs, verbose=False,
                use_ls=use_ls, use_pe=use_pe, use_rn=use_rn,
                hops=hops, seed=seed
            )
            
            # Sauvegarde du 'Gold Model' si le run actuel est le meilleur[cite: 18]
            if current_run_acc > best_acc_for_task:
                best_acc_for_task = current_run_acc
                gold_path = os.path.join(save_dir, f"gold_model_task{task_id}.pth")
                temp_path = os.path.join(save_dir, f"best_model_task{task_id}.pth")
                
                if os.path.exists(temp_path):
                    shutil.copyfile(temp_path, gold_path)
                print(f"    New Absolute Best: {best_acc_for_task:.4f}")

        results[task_id] = best_acc_for_task

    print("\nFINAL BEST-OF-N SUMMARY")
    for tid, acc in results.items():
        print(f"Task {tid:02d}: {acc:.4f} accuracy")

def run_joint_best(n_runs=5, use_ls=True, use_pe=True, use_rn=False, hops=3, epochs=60):
    """Exécute l'entraînement Joint plusieurs fois pour garder le meilleur modèle global."""
    print(f"Starting Joint training ({n_runs} attempts to find the best global model)...\n")
    
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    
    best_mean_acc = -1.0
    
    for run in range(n_runs):
        seed = run
        print(f"Joint Attempt {run+1}/{n_runs} (Seed: {seed})")
        
        # Entraînement Joint sur les 20 tâches
        per_task_acc = train_joint(
            epochs=epochs, verbose=True,
            use_ls=use_ls, use_pe=use_pe, use_rn=use_rn,
            hops=hops, seed=seed
        )
        
        mean_acc = sum(per_task_acc.values()) / 20
        
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            shutil.copyfile(os.path.join(save_dir, "best_model_joint.pth"), 
                            os.path.join(save_dir, "gold_model_joint.pth"))
            print(f"  New Best Joint Mean Accuracy: {best_mean_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", action="store_true", help="Enable Joint training instead of per-task")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per task or joint attempts")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if args.joint:
        run_joint_best(n_runs=args.runs, epochs=args.epochs)
    else:
        run_best_of_n(n_runs=args.runs, epochs=args.epochs)