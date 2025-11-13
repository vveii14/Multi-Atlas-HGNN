import os
import json
import pdb
import torch
import datetime
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from multiprocessing import Pool, cpu_count, set_start_method
import warnings
warnings.filterwarnings("ignore")  # 屏蔽所有警告

from utils import * 
from model import *


def train(model, args: Args, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    total_loss = 0
    for data in train_loader:  
        data = data.to(args.device)
        # print('data:', data)
        out = model(data) 
        loss = criterion(out, data.y) 
        total_loss +=loss
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
    return total_loss/len(train_loader.dataset)


@torch.no_grad()
def test(model, args: Args, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(args.device)
            out = model(data)
            probs = F.softmax(out, dim=1)  # Calculate probabilities
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy()[:, 1])  # Keep the probabilities of the positive class
            all_labels.append(data.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    metrics = {
        'accuracy': accuracy,
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1
    }
    
    return metrics


def train_and_evaluate_fold(args, train_loader, val_loader, test_loader, fold_idx, verbose=False):
    """
    Train and evaluate model for a single fold
    
    Args:
        args: Arguments object
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        fold_idx: Fold index
        verbose: Whether to print detailed information
    
    Returns:
        dict: Test metrics for this fold
    """
    
    checkpoints_dir = './checkpoints/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # 建议：args.model 直接用字符串控制
    if args.model == "HGCNConv":
        model = ResidualHyperGNNs(
            args, train_loader.dataset, args.hidden, args.hidden_mlp, args.num_layers
        ).to(args.device)
    else:
        gnn = eval(args.model)  # "GINConv"/"ChebConv"/"GCNConv" 等
        model = ResidualGNNs(
            args, train_loader.dataset, args.hidden, args.hidden_mlp, args.num_layers, gnn
        ).to(args.device)

    if verbose:
        print(model)
        
    # Training loop
    best_val_auroc = 0.0
    val_acc_history, test_acc_history, test_loss_history = [], [], []
    
    for epoch in tqdm(range(args.epochs), desc=f"Fold {fold_idx + 1}", position=args.rank + 1, leave=True):
        # Train
        train_loss = train(model, args, train_loader)
        
        # Validate
        val_metrics = test(model, args, val_loader)
        
        if verbose:
            train_metrics = test(model, args, train_loader)
            test_metrics = test(model, args, test_loader)
            print(f"Epoch {epoch}: Loss={train_loss:.6f}, Val_AUROC={val_metrics['auroc']:.4f}, Test_AUROC={test_metrics['auroc']:.4f}")
        
        # Save best model based on validation AUROC
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), f"{checkpoints_dir}{args.mode}_{args.edge_dir_prefix.split('/')[0]}_{args.model}{args.tune_name}_fold{fold_idx+1}_best.pkl")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f"{checkpoints_dir}{args.mode}_{args.edge_dir_prefix.split('/')[0]}_{args.model}{args.tune_name}_fold{fold_idx+1}_best.pkl"))
    model.eval()
    test_metrics = test(model, args, test_loader)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return test_metrics


def _fold_progress_file(args):
    checkpoints_dir = './checkpoints/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    resume_filename = f"{args.mode}_{args.edge_dir_prefix.split('/')[0]}_{args.model}{args.tune_name}_fold_progress.json"
    return os.path.join(checkpoints_dir, resume_filename)


def _load_fold_progress(args):
    progress = {"completed_folds": {}}
    path = _fold_progress_file(args)
    if not os.path.exists(path):
        return progress
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("completed_folds"), dict):
            progress["completed_folds"] = {
                k: v for k, v in data["completed_folds"].items() if isinstance(v, dict)
            }
    except (OSError, json.JSONDecodeError):
        pass
    return progress


def _save_fold_progress(args, progress):
    path = _fold_progress_file(args)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f)
    os.replace(tmp_path, path)


def _sanitize_metrics(metrics):
    return {key: float(val) for key, val in metrics.items()}


def predefined_5fold_cross_validation(args, verbose=False):
    """
    Perform 5-fold cross validation using predefined fold assignments
    
    Args:
        args: Arguments object
        verbose: Whether to print detailed information
    
    Returns:
        tuple: (avg_metrics, std_metrics)
    """
    
    # Define the 5 fold combinations (fixed rotation: 3 train, 1 val, 1 test)
    fold_combinations = [
        {'train_folds': [1, 2, 3], 'val_fold': 4, 'test_fold': 5},
        {'train_folds': [2, 3, 4], 'val_fold': 5, 'test_fold': 1},
        {'train_folds': [3, 4, 5], 'val_fold': 1, 'test_fold': 2},
        {'train_folds': [4, 5, 1], 'val_fold': 2, 'test_fold': 3},
        {'train_folds': [5, 1, 2], 'val_fold': 3, 'test_fold': 4},
    ]
    
    progress = _load_fold_progress(args)
    completed_folds = progress.get("completed_folds", {})
    fold_metrics = []
    fold_failed = False
    
    for fold_idx, fold_info in enumerate(fold_combinations):
        fold_key = str(fold_idx + 1)
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/5: Train={fold_info['train_folds']}, Val={fold_info['val_fold']}, Test={fold_info['test_fold']}")
        print(f"{'='*60}")
        if fold_key in completed_folds:
            metrics = _sanitize_metrics(completed_folds[fold_key])
            completed_folds[fold_key] = metrics
            fold_metrics.append(metrics)
            print(f"[INFO] Fold {fold_idx + 1} already completed. Skipping training.")
            continue
        
        try:
            # Load data for this fold combination
            train_data, val_data, test_data, train_labels, val_labels, test_labels = load_data_with_fold_split_ori(args, fold_info)
            
            if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
                print(f"[WARNING] Empty dataset in fold {fold_idx + 1}. Skipping.")
                continue
            
            # Create data loaders
            train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_data, args.batch_size, shuffle=False, drop_last=False)
            test_loader = DataLoader(test_data, args.batch_size, shuffle=False, drop_last=False)
            
            # Train model for this fold (function moved to train.py)
            fold_metrics_result = train_and_evaluate_fold(args, train_loader, val_loader, test_loader, fold_idx, verbose)
            sanitized_metrics = _sanitize_metrics(fold_metrics_result)
            fold_metrics.append(sanitized_metrics)
            completed_folds[fold_key] = sanitized_metrics
            progress["completed_folds"] = completed_folds
            _save_fold_progress(args, progress)
            
            # Clean up memory
            del train_data, val_data, test_data, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()
            
        except Exception as e:
            fold_failed = True
            print(f"[ERROR] Failed to process fold {fold_idx + 1}: {e}")
            continue
    
    if not fold_metrics:
        raise ValueError("No valid folds were processed!")
    
    pending_folds = [idx for idx in range(1, len(fold_combinations) + 1) if str(idx) not in completed_folds]
    if fold_failed and pending_folds:
        raise RuntimeError(f"Cross validation interrupted before completing folds {pending_folds}. Resolve the error and rerun to resume.")
    
    ordered_metrics = [completed_folds[str(idx)] for idx in range(1, len(fold_combinations) + 1) if str(idx) in completed_folds]
    
    # Calculate average metrics
    avg_metrics = {key: np.mean([fold[key] for fold in ordered_metrics]) for key in ordered_metrics[0].keys()}
    std_metrics = {key: np.std([fold[key] for fold in ordered_metrics]) for key in ordered_metrics[0].keys()}
    
    if not pending_folds:
        resume_path = _fold_progress_file(args)
        if os.path.exists(resume_path):
            try:
                os.remove(resume_path)
            except OSError:
                pass
    
    if verbose:
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Average Metrics: {avg_metrics}")
        print(f"Std Metrics: {std_metrics}")
    
    return avg_metrics, std_metrics


if_method_has_constructed = []
def bench_from_args(args: Args, verbose = False, use_predefined_folds=False):

    method = args.dataset + args.mode + args.atlas + args.edge_dir_prefix
    # construct the dataset
    if method in if_method_has_constructed:
        print(f"Graph construction for {args.dataset}_{args.mode} with {args.edge_dir_prefix} already completed.")
    else:
        construct_graphs_for_all_subjects(
            args.dataset,
            args.atlas,
            args.edge_dir_prefix,
            args.rank,
            input_data_dir=args.timeseries_dir,
            mode=args.mode,
            k_hyperedges=args.k,
        )
        if_method_has_constructed.append(method)

    # Use predefined folds if requested
    if use_predefined_folds:
        print("[INFO] Using predefined 5-fold cross validation")
        avg_metrics, std_metrics = predefined_5fold_cross_validation(args, verbose)
        log_experiment_result(args, avg_metrics, std_metrics, [], "predefined_folds")
        return avg_metrics, std_metrics


# Configuration: Set to True to use predefined folds (5 fold cross我预定义的), False for random folds (随机划分的5fold-cross)
USE_PREDEFINED_FOLDS = True  # Change this to False if you want random folds

BASE_DIR = os.path.abspath(os.path.dirname(__file__)) #获取当前文件的绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) #BASE_DIR的上一层目录

argsDictTune_a = {
    'dataset' : ['ADHD'],
    'mode' : ["f_hyper_k5"],
    'timeseries_dir':'/data/public_dataset/ADHD_parcellation/fMRIPrep_pkl',
    'dataset_dir' : os.path.join(BASE_DIR, "data"),
    'folds_dir': PROJECT_ROOT,
    # choose from: GCNConv, GINConv, SGConv, GeneralConv, GATConv, HGCNConv
    'edge_dir_prefix' : [
        'hyper_pearson_correlation',
        #'pearson_correlation',
        #'cosine_similarity',
        #'euclidean_distance',
        #'spearman_correlation',
        #'kendall_correlation',
        #'partial_correlation',
        #'cross_correlation',
        #'correlations_correlation',
        #'associated_high_order_fc',
        #'knn_graph',
        # 'mutual_information',
        # 'granger_causality',
        # 'coherence_matrix',
        # 'generalised_synchronisation_matrix',
        #'patels_conditional_dependence_measures_kappa',
        #'patels_conditional_dependence_measures_tau',
    ],
    'atlas' : ["AAL116"],
    # 'atlas' : "Schaefer100",
    'model' : "HGCNConv" ,
    'num_classes' : 2,  
    #'weight_decay' : 0.0005,
    'weight_decay' : [0, 5e-4],
    #'batch_size': 32,
    'batch_size': [16, 32, 64],
    'hidden_mlp' : 64,
    #'hidden' : 32,
    'hidden': [16, 32, 64],
    'num_layers' : [2],
    # 'num_layers': 5,
    'runs' : 1,
    'lr': [5e-4, 1e-3, 5e-3],
    #'lr': 1e-3,
    #'epochs' : [20, 100],
    'epochs' : [100, 500],
    'edge_percent_to_keep': [0.1, 0.2, 0.3],
    #'edge_percent_to_kep' : [0.2],
    'n_splits' : 5,
    'seed' : 42,
    'graph_type' : 'static',
    'k' : [5],
    # 'verbose' : True
    
}

args_list_a = Args.tuning_list(argsDictTune_a)
fix_seed(args_list_a[0].seed)

#restart
existing_result_keys = set()
result_file = f"result_{args_list_a[0].mode}.txt"

if os.path.exists(result_file):
    with open(result_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("_dataset"):
                existing_result_keys.add(line)

args_list_a = [a for a in args_list_a if a.tune_name not in existing_result_keys]

gpu_count = torch.cuda.device_count()
available_gpu_ids = list(range(gpu_count)) if gpu_count > 0 else []

# ✅ 给每个任务分配一个轮询的 GPU 和 tqdm rank
for i, args in enumerate(args_list_a):
    if len(available_gpu_ids) > 0:
        args.gpu_id = available_gpu_ids[i % len(available_gpu_ids)]
        args.device = f"cuda:{args.gpu_id}"
    else:
        args.gpu_id = None
        args.device = "cpu"
    args.rank = i

# ✅ 单个任务执行函数
def run_single_experiment(args: Args, use_predefined_folds=False):
    if torch.cuda.is_available() and (args.gpu_id is not None):
        torch.cuda.set_device(args.gpu_id)
    print(f"[INFO] Running {args.tune_name} on {args.device}")
    if use_predefined_folds:
        print(f"[INFO] Using predefined folds for {args.tune_name}")
    fix_seed(args.seed)
    met, std = bench_from_args(args, verbose=True, use_predefined_folds=use_predefined_folds)

    with open(f"result_{args.mode}.txt", "a") as f:
        f.write(f"{args.tune_name}\n{met}\n--------------------------------\n")
    print(f"✅ {args.tune_name} done on {args.device}")
    return met

# ✅ 控制要开多少任务
max_parallel_tasks = max(1, gpu_count)   # 你最多可以设置为 cpu_count()


# ✅ 启动任务池
if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    print(f"[INFO] Using {'predefined' if USE_PREDEFINED_FOLDS else 'random'} folds")
    
    with Pool(processes=min(max_parallel_tasks, cpu_count())) as pool:
        # Pass the USE_PREDEFINED_FOLDS flag to each experiment
        test_metric_list_a = pool.starmap(run_single_experiment, [(args, USE_PREDEFINED_FOLDS) for args in args_list_a])
