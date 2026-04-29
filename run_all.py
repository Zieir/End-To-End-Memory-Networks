from train import train_task
from eval import evaluate_task
from data.dataloader import BabiDataset

def run_all():
    dataset = BabiDataset(download=False)
    results = {}
        
    for task_id in range(1, 21):
        try:
            print(f"Processing Task {task_id:02d}: {dataset.TASK_DESCRIPTIONS[task_id]}...", end=" ", flush=True)
            
            # Reusing your code!
            train_task(task_id, epochs=100, verbose=False)
            best_acc = evaluate_task(task_id, verbose=False)
            
            error_rate = (1 - best_acc) * 100
            results[task_id] = {
                'description': dataset.TASK_DESCRIPTIONS[task_id],
                'acc': best_acc,
                'error': error_rate
            }
            print(f"Done! (Error: {error_rate:.1f}%)")
            
        except Exception as e:
            print(f"FAILED! Error: {e}")
            results[task_id] = {
                'description': dataset.TASK_DESCRIPTIONS.get(task_id, "Unknown"), 
                'acc': 0.0, 
                'error': 100.0
            }

    print("FINAL RESULTS SUMMARY (BoW Model - 1k Dataset)")
    print(f"{'Task':<4} | {'Description':<25} | {'Test Acc':<10} | {'Error Rate (%)':<15}")
    
    mean_error = 0
    failed_tasks = 0
    
    for task_id in range(1, 21):
        res = results.get(task_id)
        if res:
            mean_error += res['error']
            if res['error'] > 5.0:
                failed_tasks += 1
            print(f"{task_id:<4} | {res['description']:<25} | {res['acc']:<10.4f} | {res['error']:.1f}%")
            
    print(f"Mean Error Rate: {mean_error / 20:.1f}%")
    print(f"Failed Tasks (>5% error): {failed_tasks} / 20")

if __name__ == "__main__":
    run_all()