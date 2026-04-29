import argparse
from train import train_task
from eval import evaluate_task
from data.dataloader import BabiDataset

def run_all(use_ls=False, use_pe=False, epochs=100, seed=0):
    dataset = BabiDataset(download=False)
    results = {}

    tag_parts = []
    tag_parts.append("PE" if use_pe else "BoW")
    if use_ls:
        tag_parts.append("LS")
    tag = "+".join(tag_parts)
    print(f"Running all 20 tasks ({tag}, epochs={epochs}, seed={seed})\n")

    for task_id in range(1, 21):
        try:
            print(f"Processing Task {task_id:02d}: {dataset.TASK_DESCRIPTIONS[task_id]}...",
                  end=" ", flush=True)

            train_task(task_id, epochs=epochs, verbose=False,
                       use_ls=use_ls, use_pe=use_pe, seed=seed)
            best_acc = evaluate_task(task_id, verbose=False)

            error_rate = (1 - best_acc) * 100
            results[task_id] = {
                'description': dataset.TASK_DESCRIPTIONS[task_id],
                'acc': best_acc,
                'error': error_rate,
            }
            print(f"Done! (Error: {error_rate:.1f}%)")

        except Exception as e:
            print(f"FAILED! Error: {e}")
            results[task_id] = {
                'description': dataset.TASK_DESCRIPTIONS.get(task_id, "Unknown"),
                'acc': 0.0,
                'error': 100.0,
            }

    print(f"\nFINAL RESULTS SUMMARY ({tag} + Temporal, 1k Dataset)")
    print(f"{'Task':<4} | {'Description':<25} | {'Test Acc':<10} | {'Error Rate (%)':<15}")

    mean_error = 0
    failed_tasks = 0

    for task_id in range(1, 21):
        res = results.get(task_id)
        if res:
            mean_error += res['error']
            if res['error'] > 5.0:
                failed_tasks += 1
            print(f"{task_id:<4} | {res['description']:<25} | "
                  f"{res['acc']:<10.4f} | {res['error']:.1f}%")

    print(f"Mean Error Rate: {mean_error / 20:.1f}%")
    print(f"Failed Tasks (>5% error): {failed_tasks} / 20")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ls", action="store_true", help="Enable Linear Start training")
    parser.add_argument("--pe", action="store_true", help="Enable Position Encoding")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_all(use_ls=args.ls, use_pe=args.pe, epochs=args.epochs, seed=args.seed)
