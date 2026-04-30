import argparse
from train_miniBatch import train_task, train_joint
from eval import evaluate_task
from data.dataloader import BabiDataset

def _print_summary(tag, results):
    print(f"\nFINAL RESULTS SUMMARY ({tag} + Temporal, 1k Dataset)")
    print(f"{'Task':<4} | {'Description':<25} | {'Test Acc':<10} | {'Error Rate (%)':<15}")

    mean_error = 0.0
    failed_tasks = 0
    for task_id in range(1, 21):
        res = results.get(task_id)
        if res is None:
            continue
        mean_error += res['error']
        if res['error'] > 5.0:
            failed_tasks += 1
        print(f"{task_id:<4} | {res['description']:<25} | "
              f"{res['acc']:<10.4f} | {res['error']:.1f}%")
    print(f"Mean Error Rate: {mean_error / 20:.1f}%")
    print(f"Failed Tasks (>5% error): {failed_tasks} / 20")

def run_per_task(use_ls=False, use_pe=False, use_rn=False, hops=3, epochs=100, seed=0,
                 tying='adjacent', use_relu=False):
    dataset = BabiDataset(download=False)
    results = {}

    tag_parts = ["PE" if use_pe else "BoW"]
    if use_ls:
        tag_parts.append("LS")
    if use_rn:
        tag_parts.append("RN")
    if tying == 'layerwise':
        tag_parts.append("LW")
    if use_relu:
        tag_parts.append("†")
    tag_parts.append(f"K={hops}")
    tag = "+".join(tag_parts)
    print(f"Running all 20 tasks per-task ({tag}, epochs={epochs}, seed={seed})\n")

    for task_id in range(1, 21):
        try:
            print(f"Processing Task {task_id:02d}: {dataset.TASK_DESCRIPTIONS[task_id]}...",
                  end=" ", flush=True)
            train_task(task_id, epochs=epochs, verbose=False,
                       use_ls=use_ls, use_pe=use_pe, use_rn=use_rn,
                       hops=hops, seed=seed,
                       tying=tying, use_relu=use_relu)
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

    _print_summary(tag, results)

def run_joint(use_ls=False, use_pe=False, use_rn=False, hops=3, epochs=60, seed=0,
              tying='adjacent', use_relu=False):
    dataset = BabiDataset(download=False)

    tag_parts = ["PE" if use_pe else "BoW"]
    if use_ls:
        tag_parts.append("LS")
    if use_rn:
        tag_parts.append("RN")
    if tying == 'layerwise':
        tag_parts.append("LW")
    if use_relu:
        tag_parts.append("†")
    tag_parts.append(f"K={hops}")
    tag_parts.append("joint")
    tag = "+".join(tag_parts)
    print(f"Joint training over all 20 tasks ({tag}, epochs={epochs}, seed={seed})\n")

    _, per_task_acc = train_joint(epochs=epochs, verbose=True,
                                   use_ls=use_ls, use_pe=use_pe, use_rn=use_rn,
                                   hops=hops, seed=seed,
                                   tying=tying, use_relu=use_relu)

    results = {}
    for task_id in range(1, 21):
        acc = per_task_acc.get(task_id, 0.0) if per_task_acc else 0.0
        results[task_id] = {
            'description': dataset.TASK_DESCRIPTIONS[task_id],
            'acc': acc,
            'error': (1 - acc) * 100,
        }
    _print_summary(tag, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ls", action="store_true", help="Enable Linear Start training")
    parser.add_argument("--pe", action="store_true", help="Enable Position Encoding")
    parser.add_argument("--rn", action="store_true", help="Enable Random Noise on temporal indices")
    parser.add_argument("--lw", action="store_true",
                        help="Use layer-wise (RNN-like) weight tying with learned H matrix "
                             "instead of adjacent tying")
    parser.add_argument("--relu", action="store_true",
                        help="Apply ReLU to half the units of the internal state after each hop "
                             "(Appendix A † variant)")
    parser.add_argument("--joint", action="store_true", help="Joint training over all 20 tasks (shared model)")
    parser.add_argument("--hops", type=int, default=3, help="Number of memory hops K")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tying = 'layerwise' if args.lw else 'adjacent'

    if args.joint:
        epochs = args.epochs if args.epochs is not None else 60
        run_joint(use_ls=args.ls, use_pe=args.pe, use_rn=args.rn,
                  hops=args.hops, epochs=epochs, seed=args.seed,
                  tying=tying, use_relu=args.relu)
    else:
        epochs = args.epochs if args.epochs is not None else 100
        run_per_task(use_ls=args.ls, use_pe=args.pe, use_rn=args.rn,
                     hops=args.hops, epochs=epochs, seed=args.seed,
                     tying=tying, use_relu=args.relu)
