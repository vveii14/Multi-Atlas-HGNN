import os
import json
import pdb
import torch
import random
import datetime
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import scipy.io as sio
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import aggr
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, SGConv, GeneralConv
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch.nn import Conv1d, MaxPool1d, ModuleList
from utils_graph_construction import *


def construct_graphs_for_all_subjects(dataset: str, atlas: str, method: str,
                                      tqdm_position: int = 0,
                                      input_data_dir="./data/fMRIROItimeseries",
                                      base_results_dir="./data/graphconstructionedge", mode="", k_hyperedges=10):
    """
    Construct graphs for all subjects given dataset, atlas, and method.
    Adds a tqdm progress bar.
    """
    method_func = {
        "pearson_correlation": pearson_correlation,
        "cosine_similarity": cosine_similarity,
        "euclidean_distance": euclidean_distance,
        "spearman_correlation": spearman_correlation,
        "kendall_correlation": kendall_correlation,
        "partial_correlation": partial_correlation,
        "cross_correlation": cross_correlation,
        "mutual_information": mutual_information,
        "correlations_correlation": correlations_correlation,
        "associated_high_order_fc": associated_high_order_fc,
        "knn_graph": knn_graph,
        "granger_causality": granger_causality,
        "coherence_matrix": coherence_matrix,
        "generalised_synchronisation_matrix": generalised_synchronisation_matrix,
        "patels_conditional_dependence_measures_kappa": patels_conditional_dependence_measures_kappa,
        "patels_conditional_dependence_measures_tau": patels_conditional_dependence_measures_tau,

        'hyper_pearson_correlation': pearson_correlation,
        'distance_correlation': pearson_correlation,
        'combine_pearson_correlation': pearson_correlation,
        'ts_pearson_correlation': pearson_correlation,
        'ts_roi_pearson_correlation': pearson_correlation,

        #"hyper_correlation": hyper_correlation,
    }.get(method, None)

    if method_func is None:
        raise ValueError(f"Unsupported method: {method}")

    candidate_folders = [
        # Prefer dataset/atlas layout but fall back to a flat atlas directory if needed
        os.path.join(input_data_dir, dataset, atlas),
        os.path.join(input_data_dir, atlas),
    ]
    input_folder = next((path for path in candidate_folders if os.path.exists(path)), None)

    if input_folder is None:
        search_paths = ", ".join(candidate_folders)
        raise FileNotFoundError(f"Input folder not found. Checked: {search_paths}")

    output_folder = os.path.join(base_results_dir, "static", dataset, mode, atlas, method)
    os.makedirs(output_folder, exist_ok=True)

    pattern = f"dataset-{dataset}_sub-"
    file_list = [
        f for f in os.listdir(input_folder)
        if f.startswith(pattern) and f.endswith(f"_desc-fMRIROItimeseries_atlas-{atlas}.pkl")
    ]

    tasks = []
    for filename in file_list:
        subject_id = filename.split("_sub-")[1].split("_")[0]
        out_filename = f"dataset-{dataset}_sub-{subject_id}_task-rest_desc-staticgraphconstructionedge_atlas-{atlas}_contrmethd-{method}.pkl"
        out_path = os.path.join(output_folder, out_filename)
        if os.path.exists(out_path):
            continue
        tasks.append((filename, subject_id, out_path))

    if not tasks:
        print(f"[INFO] Graphs already exist for {dataset}-{atlas}-{method}. Skipping construction.")
        return

    for filename, subject_id, out_path in tqdm(tasks, desc=f"Constructing graphs ({dataset}-{mode}-{atlas}-{method})", leave=True, position=tqdm_position):
        input_filepath = os.path.join(input_folder, filename)

        data = pd.read_pickle(input_filepath)
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            corr = method_func(data)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        if method == 'distance_correlation':
            data_list = [construct_H_with_KNN_from_distance(feature = data, k_neig = k_hyperedges)]
            data_batch = Batch.from_data_list(data_list)
        elif method == 'hyper_pearson_correlation':
            data_list = [correlation_matrix_to_hypergraph_data(corr, k_neig=k_hyperedges)]
            data_batch = Batch.from_data_list(data_list)
        elif method == 'combine_pearson_correlation':
            data_list = [combine_correlation_matrix_to_hypergraph_data(corr, k_neig=k_hyperedges)]
            data_batch = Batch.from_data_list(data_list)
        elif method == 'pearson_correlation':
            data_list = [correlation_matrix_to_graph_data(corr)]
            data_batch = Batch.from_data_list(data_list)
        elif method == 'ts_pearson_correlation':
            data_list = [timeseries_correlation_matrix_to_graph_data(corr, data, k_neig=k_hyperedges)]
            data_batch = Batch.from_data_list(data_list)
        elif method == 'ts_roi_pearson_correlation':
            data_list = [timeseries_roi_correlation_matrix_to_graph_data(corr, data, k_neig=k_hyperedges)]
            data_batch = Batch.from_data_list(data_list)
        
        torch.save(data_batch, out_path)

# Patterned data generator
def generate_complex_patterned_data(num_samples, num_rois, time_steps):
    X = torch.zeros(num_samples, num_rois, time_steps)
    y = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        for roi in range(num_rois):
            # Mix of sinusoidal, linear, and random noise patterns
            if i < num_samples // 2:
                # Class 0: Mix of patterns with more emphasis on sinusoidal
                signal = (
                    0.6 * torch.sin(torch.linspace(0, 2 * np.pi, time_steps)) +
                    0.3 * torch.linspace(0, 1, time_steps) +
                    0.1 * torch.randn(time_steps)
                )
            else:
                # Class 1: Mix of patterns with more emphasis on linear
                signal = (
                    0.3 * torch.sin(torch.linspace(0, 2 * np.pi, time_steps)) +
                    0.6 * torch.linspace(0, 1, time_steps) +
                    0.1 * torch.randn(time_steps)
                )
            X[i, roi, :] = signal

        y[i] = 0 if i < num_samples // 2 else 1

    return X, y


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Helper function from one dictionary to multiple sets of params
def grid_from_param(dic): #, append_model_name = True, model_name_default = 'default'):
    # retrieve all lists to choose parameters from
    names = []
    lens = []
    for name in dic:
        item = dic[name]
        if(type(item)==list):
            names.append(name)
            lens.append(len(item))

    #helper
    count = 1
    mults = []
    for l in lens:
        mults.append(count)
        count *= l

    #construct the grid
    params = []
    n = len(lens)
    for i in range(count):
        param = dic.copy()
        param['tune_name'] = '_'
        for j in range(n):
            param[names[j]] = dic[names[j]][i//mults[j] % lens[j]]
            param['tune_name'] += f"{names[j]}{param[names[j]]}_"
            # if param['tune_name'] contains '/', replace with '_'
            param['tune_name'] = param['tune_name'].replace('/', '_')
        params.append(param)
    return params

class Args: # now it's just a wrapper for compatibility. Everything now packed up in the dictionary.
    def __init__(self, param_dict) -> None:
        # wrapped. see argDict above.
        self.dataset = param_dict['dataset']
        self.dataset_dir = param_dict['dataset_dir']
        self.timeseries_dir = param_dict.get('timeseries_dir', "./data/fMRIROItimeseries")
        self.folds_dir = param_dict.get('folds_dir', self.dataset_dir)
        self.edge_dir_prefix = param_dict['edge_dir_prefix']
        self.model = param_dict['model']
        self.num_classes = param_dict['num_classes']
        self.weight_decay = param_dict['weight_decay']
        self.batch_size = param_dict['batch_size']
        self.hidden_mlp = param_dict['hidden_mlp']
        self.hidden = param_dict['hidden']
        self.num_layers = param_dict['num_layers']
        self.runs = param_dict['runs']
        self.lr = param_dict['lr']
        self.epochs = param_dict['epochs']
        self.edge_percent_to_keep = param_dict['edge_percent_to_keep'] 
        self.seed = param_dict['seed']
        self.n_splits = param_dict['n_splits'] if "n_splits" in param_dict else 5
        self.device = "cuda" if self.model != "GATConv" else "cpu"
        self.tune_name = param_dict['tune_name'] if "tune_name" in param_dict else None
        self.atlas = param_dict['atlas'] if "atlas" in param_dict else None
        self.graph_type = param_dict['graph_type'] if "graph_type" in param_dict else None
        self.rank = param_dict['rank'] if "rank" in param_dict else None
        # add hyper flag (was missing) so code can access args.hyper
        self.mode = param_dict.get('mode', "")
        self.k = param_dict.get('k', 10)

    def tuning_list(param_dicts : dict):
        p = grid_from_param(param_dicts)
        return [Args(x) for x in p]


def load_data_from_args(args, fold_info=None):
    """
    Load data from args with optional fold information
    
    Args:
        args: Arguments object
        fold_info: Dictionary with 'train_folds', 'val_fold', 'test_fold' keys
                  e.g., {'train_folds': [1,2,3], 'val_fold': 4, 'test_fold': 5}
    """
    
    # Load fold assignments if provided
    if fold_info is not None:
        fold_assignments_file = os.path.join(args.folds_dir, '5_folds', 'fold_assignments.csv')
        if os.path.exists(fold_assignments_file):
            fold_df = pd.read_csv(fold_assignments_file, dtype={'IID': str})
            print(f"[INFO] Using predefined fold assignments from {fold_assignments_file}")
        else:
            print(f"[WARNING] Fold assignments file not found: {fold_assignments_file}")
            fold_info = None
    
    # Load labels
    if fold_info is not None:
        # Use fold assignments to filter subjects
        train_folds = fold_info['train_folds']
        val_fold = fold_info['val_fold'] 
        test_fold = fold_info['test_fold']
        
        # Get IIDs for each fold
        train_iids = fold_df[fold_df['fold'].isin(train_folds)]['IID'].tolist()
        val_iids = fold_df[fold_df['fold'] == val_fold]['IID'].tolist()
        test_iids = fold_df[fold_df['fold'] == test_fold]['IID'].tolist()
        
        # Combine all IIDs for this experiment
        all_iids = train_iids + val_iids + test_iids
        
        # Create labels dataframe
        labels_df = fold_df[fold_df['IID'].isin(all_iids)].copy()
        labels_df = labels_df[['IID', 'Diagnosis']].reset_index(drop=True)
        
        print(f"[INFO] Fold {fold_info['train_folds']} (train): {len(train_iids)} subjects")
        print(f"[INFO] Fold {val_fold} (val): {len(val_iids)} subjects") 
        print(f"[INFO] Fold {test_fold} (test): {len(test_iids)} subjects")
    else:
        # Original logic - load from y.csv
        labels_file = os.path.join(args.dataset_dir, 'y', args.dataset, 'y.csv')
        labels_df = pd.read_csv(labels_file, dtype={'IID': str})

    if args.dataset == 'ADNI':
        label0, label1 = 0, 4
    elif args.dataset == 'HCP' or args.dataset == 'ADHD':
        label0, label1 = 0, 1

    labels_df = labels_df[labels_df['Diagnosis'].isin([label0, label1])].reset_index(drop=True)
    # and change the labels to 0 and 1
    labels_df['Diagnosis'] = labels_df['Diagnosis'].replace({label0: 0, label1: 1})
    print(f"[INFO] Number of samples after label filtering: {len(labels_df)}")

    dataset = []
    all_edge_counts = []
    corrupted_iids = []  # ← 记录坏图 ID
    missing_iids = []    # ← 记录缺失图的 ID

    graph_dir = os.path.join(args.dataset_dir, 'graphconstructionedge', args.graph_type, args.dataset, args.atlas, args.edge_dir_prefix)
    # First pass: gather edge count info
    for i in range(len(labels_df)):
        IID = str(labels_df['IID'][i])
        graph_filename = f'dataset-{args.dataset}_sub-{IID}_task-rest_desc-staticgraphconstructionedge_atlas-{args.atlas}_contrmethd-{args.edge_dir_prefix}.pkl'
        graph_path = os.path.join(graph_dir, graph_filename)

        if not os.path.exists(graph_path):
            missing_iids.append(IID)
            continue

        try:
            # Files are trusted in this environment: allow full unpickling (weights_only=False)
            data_batch = torch.load(graph_path, map_location='cpu', weights_only=False)
            data = data_batch[0]
            edge_attr = data.x.numpy()
            if edge_attr.shape[0] != edge_attr.shape[1]:
                print(f'[WARNING] Non-square matrix in {IID}, shape={edge_attr.shape}. Skipping.')
                corrupted_iids.append(IID)
                continue
            if np.isnan(edge_attr).any() or np.isinf(edge_attr).any():
                print(f'[CORRUPTED] NaN or Inf in edge_attr (first pass) — {IID}. Skipping.')
                corrupted_iids.append(IID)
                continue
            np.fill_diagonal(edge_attr, 0)
            edge_index = np.vstack(np.nonzero(edge_attr))
            all_edge_counts.append(edge_index.shape[1])
        except Exception as e:
            print(f'[ERROR] Failed to load {graph_path}: {e}')
            corrupted_iids.append(IID)
            continue

    if not all_edge_counts:
        raise ValueError("[FATAL] No valid graphs were loaded. Please check dataset_dir and file naming.")

    fixed_edge_count = min(all_edge_counts)
    print(f"[INFO] Fixed edge count set to {fixed_edge_count}")

    # Second pass: build data objects
    for i in range(len(labels_df)):
        IID = str(labels_df['IID'][i])
        y = torch.tensor(labels_df['Diagnosis'][i], dtype=torch.long)
        graph_filename = f'dataset-{args.dataset}_sub-{IID}_task-rest_desc-staticgraphconstructionedge_atlas-{args.atlas}_contrmethd-{args.edge_dir_prefix}.pkl'
        graph_path = os.path.join(graph_dir, graph_filename)

        if not os.path.exists(graph_path):
            if IID not in missing_iids:
                missing_iids.append(IID)
            continue
        if IID in corrupted_iids:
            continue

        try:
            # Files are trusted in this environment: allow full unpickling (weights_only=False)
            data_batch = torch.load(graph_path, map_location='cpu', weights_only=False)
            data_obj = data_batch[0]
            edge_attr = data_obj.x.numpy()
            if edge_attr.shape[0] != edge_attr.shape[1]:
                print(f'[WARNING] Non-square matrix in {IID}, shape={edge_attr.shape}. Skipping.')
                corrupted_iids.append(IID)
                continue
            if np.isnan(edge_attr).any() or np.isinf(edge_attr).any():
                print(f'[CORRUPTED] NaN or Inf in edge_attr (second pass) — {IID}. Skipping.')
                corrupted_iids.append(IID)
                continue
            np.fill_diagonal(edge_attr, 0)
        except Exception as e:
            print(f'[ERROR] Failed to load {graph_path}: {e}')
            corrupted_iids.append(IID)
            continue

        # Thresholding
        threshold = np.percentile(edge_attr, 100 * (1 - args.edge_percent_to_keep))
        edge_attr[edge_attr <= threshold] = 0

        edge_index = np.vstack(np.nonzero(edge_attr))
        filtered_edge_attr = edge_attr[edge_index[0], edge_index[1]]
        filtered_edge_attr = torch.tensor(filtered_edge_attr, dtype=torch.float)

        current_edge_count = edge_index.shape[1]
        if current_edge_count > fixed_edge_count:
            sorted_indices = torch.argsort(filtered_edge_attr, descending=True)
            indices_to_keep = sorted_indices[:fixed_edge_count]
            edge_index = edge_index[:, indices_to_keep]
            filtered_edge_attr = filtered_edge_attr[indices_to_keep]
        elif current_edge_count < fixed_edge_count:
            sorted_indices = torch.argsort(filtered_edge_attr, descending=False)
            indices_to_add = sorted_indices[:fixed_edge_count - current_edge_count]
            edge_index_to_add = edge_index[:, indices_to_add]
            filtered_edge_attr_to_add = filtered_edge_attr[indices_to_add]
            edge_index = np.hstack([edge_index, edge_index_to_add])
            filtered_edge_attr = torch.cat([filtered_edge_attr, filtered_edge_attr_to_add])

        data = Data(
            x=torch.tensor(edge_attr, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=filtered_edge_attr,
            y=y
        )

        dataset.append(data)

    # 最后打印缺失和异常 IID
    if missing_iids:
        print(f"\nSkipped {len(missing_iids)} IIDs due to not found these file.")
    if corrupted_iids:
        print(f"\nSkipped {len(corrupted_iids)} IIDs due to data errors.")
        print(corrupted_iids)
    elif not missing_iids:
        print("\n✅ No corrupted graphs found.")

    return dataset


def load_data_with_fold_split(args, fold_info):
    """
    Load data and split into train/val/test based on fold information
    
    Args:
        args: Arguments object
        fold_info: Dictionary with 'train_folds', 'val_fold', 'test_fold' keys
    
    Returns:
        tuple: (train_data, val_data, test_data, train_labels, val_labels, test_labels)
    """
    
    # Load fold assignments
    fold_assignments_file = os.path.join(args.folds_dir, '5_folds', 'fold_assignments.csv')
    if not os.path.exists(fold_assignments_file):
        raise FileNotFoundError(f"Fold assignments file not found: {fold_assignments_file}")
    
    fold_df = pd.read_csv(fold_assignments_file, dtype={'IID': str})
    
    # Get IIDs for each fold
    train_folds = fold_info['train_folds']
    val_fold = fold_info['val_fold'] 
    test_fold = fold_info['test_fold']
    
    train_iids = fold_df[fold_df['fold'].isin(train_folds)]['IID'].tolist()
    val_iids = fold_df[fold_df['fold'] == val_fold]['IID'].tolist()
    test_iids = fold_df[fold_df['fold'] == test_fold]['IID'].tolist()
    
    print(f"[INFO] Fold {train_folds} (train): {len(train_iids)} subjects")
    print(f"[INFO] Fold {val_fold} (val): {len(val_iids)} subjects") 
    print(f"[INFO] Fold {test_fold} (test): {len(test_iids)} subjects")
    
    # Load data for each split
    train_data, train_labels = load_data_for_iids(args, train_iids)
    val_data, val_labels = load_data_for_iids(args, val_iids)
    test_data, test_labels = load_data_for_iids(args, test_iids)
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

def load_data_for_iids(args, iids):
    dataset = []
    labels = []
    corrupted_iids = []
    missing_iids = []

    graph_dir = os.path.join(
        args.dataset_dir,
        'graphconstructionedge',
        args.graph_type,
        args.dataset,
        args.mode,
        args.atlas,
        args.edge_dir_prefix
    )

    fold_assignments_file = os.path.join(args.folds_dir, '5_folds', 'fold_assignments.csv')
    fold_df = pd.read_csv(fold_assignments_file, dtype={'IID': str})

    for IID in iids:
        graph_filename = (
            f'dataset-{args.dataset}_sub-{IID}_task-rest_desc-staticgraphconstructionedge_'
            f'atlas-{args.atlas}_contrmethd-{args.edge_dir_prefix}.pkl'
        )
        graph_path = os.path.join(graph_dir, graph_filename)

        if not os.path.exists(graph_path):
            missing_iids.append(IID)
            continue

        try:
            data_batch = torch.load(graph_path, map_location='cpu', weights_only=False)
            data_obj = data_batch[0]

            label_row = fold_df[fold_df['IID'] == IID]
            if len(label_row) == 0:
                corrupted_iids.append(IID)
                continue

            y_val = int(label_row['Diagnosis'].iloc[0])
            if args.dataset in ['ADNI', 'HCP', 'ADHD']:
                y_val = 0 if y_val == 0 else 1
            y = torch.tensor(y_val, dtype=torch.long)

            x = data_obj.x.float()
            H = data_obj.H.float() 

            data = Data(
                x=x,
                H=H,
                y=y
            )

            dataset.append(data)
            labels.append(y)

        except Exception as e:
            print(f'[ERROR] Failed to load {graph_path}: {e}')
            corrupted_iids.append(IID)
            continue

    if missing_iids:
        print(f"\nSkipped {len(missing_iids)} IIDs due to not found these file.")
    if corrupted_iids:
        print(f"\nSkipped {len(corrupted_iids)} IIDs due to data errors.")
        print(corrupted_iids[:10])
        if len(corrupted_iids) > 10:
            print(f"... and {len(corrupted_iids) - 10} more")

        skipped_file = (
            f"./logs/skipped_iids_{args.dataset}_{args.atlas}_"
            f"{args.edge_dir_prefix.replace('/', '_')}.txt"
        )
        os.makedirs("./logs", exist_ok=True)
        with open(skipped_file, "w") as f:
            f.write(f"Skipped IIDs for {args.dataset}_{args.atlas}_{args.edge_dir_prefix}\n")
            f.write(f"Missing files: {len(missing_iids)}\n")
            f.write(f"Total data errors: {len(corrupted_iids)}\n")
            f.write("=" * 50 + "\n")
            for iid in corrupted_iids:
                f.write(f"{iid}\n")
        print(f"[INFO] Skipped IIDs saved to: {skipped_file}")

    print(f"[INFO] Successfully loaded {len(dataset)} samples")
    return dataset, labels


def log_experiment_result(args, avg_metrics, std_metrics, fold_metrics, filename_prefix="log"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建结构化日志 dict
    log = {
        "timestamp": timestamp,
        "tune_name": args.tune_name,
        "dataset": args.dataset,
        "atlas": args.atlas,
        "graph_type": args.graph_type,
        "edge_dir_prefix": args.edge_dir_prefix,
        "model": args.model,
        "num_layers": args.num_layers,
        "hidden": args.hidden,
        "hidden_mlp": args.hidden_mlp,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "edge_percent_to_keep": args.edge_percent_to_keep,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "val_avg_metrics": avg_metrics,
        "val_std_metrics": std_metrics,
        "fold_metrics": fold_metrics,
    }

    # 创建 logs 目录
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 保存为 jsonl 文件，每一行一个实验
    log_path = os.path.join(log_dir, f"{filename_prefix}_{args.dataset}.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log) + "\n")



def load_data_with_fold_split_ori(args, fold_info):
    """
    Load data and split into train/val/test based on fold information
    
    Args:
        args: Arguments object
        fold_info: Dictionary with 'train_folds', 'val_fold', 'test_fold' keys
    
    Returns:
        tuple: (train_data, val_data, test_data, train_labels, val_labels, test_labels)
    """
    
    # Load fold assignments
    fold_assignments_file = os.path.join(args.folds_dir, '5_folds', 'fold_assignments.csv')
    if not os.path.exists(fold_assignments_file):
        raise FileNotFoundError(f"Fold assignments file not found: {fold_assignments_file}")
    
    fold_df = pd.read_csv(fold_assignments_file, dtype={'IID': str})
    
    # Get IIDs for each fold
    train_folds = fold_info['train_folds']
    val_fold = fold_info['val_fold'] 
    test_fold = fold_info['test_fold']
    
    train_iids = fold_df[fold_df['fold'].isin(train_folds)]['IID'].tolist()
    val_iids = fold_df[fold_df['fold'] == val_fold]['IID'].tolist()
    test_iids = fold_df[fold_df['fold'] == test_fold]['IID'].tolist()
    
    print(f"[INFO] Fold {train_folds} (train): {len(train_iids)} subjects")
    print(f"[INFO] Fold {val_fold} (val): {len(val_iids)} subjects") 
    print(f"[INFO] Fold {test_fold} (test): {len(test_iids)} subjects")
    
    # Load data for each split
    train_data, train_labels = load_data_for_iids_ori(args, train_iids)
    val_data, val_labels = load_data_for_iids_ori(args, val_iids)
    test_data, test_labels = load_data_for_iids_ori(args, test_iids)
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

def load_data_for_iids_ori(args, iids):
    dataset = []
    labels = []
    corrupted_iids = []
    missing_iids = []

    graph_dir = os.path.join(
        args.dataset_dir,
        'graphconstructionedge',
        args.graph_type,
        args.dataset,
        args.mode,
        args.atlas,
        args.edge_dir_prefix
    )

    fold_assignments_file = os.path.join(args.folds_dir, '5_folds', 'fold_assignments.csv')
    fold_df = pd.read_csv(fold_assignments_file, dtype={'IID': str})

    for IID in iids:
        graph_filename = (
            f'dataset-{args.dataset}_sub-{IID}_task-rest_desc-staticgraphconstructionedge_'
            f'atlas-{args.atlas}_contrmethd-{args.edge_dir_prefix}.pkl'
        )
        graph_path = os.path.join(graph_dir, graph_filename)

        if not os.path.exists(graph_path):
            missing_iids.append(IID)
            continue

        try:
            data_batch = torch.load(graph_path, map_location='cpu', weights_only=False)
            data_obj = data_batch[0]

            label_row = fold_df[fold_df['IID'] == IID]
            if len(label_row) == 0:
                corrupted_iids.append(IID)
                continue

            y_val = int(label_row['Diagnosis'].iloc[0])
            if args.dataset in ['ADNI', 'HCP', 'ADHD']:
                y_val = 0 if y_val == 0 else 1
            y = torch.tensor(y_val, dtype=torch.long)

            x = data_obj.x.float()
            edge_index = data_obj.edge_index.long()
            edge_attr = data_obj.edge_attr
            if edge_attr is not None:
                edge_attr = edge_attr.float()

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y
            )

            dataset.append(data)
            labels.append(y)

        except Exception as e:
            print(f'[ERROR] Failed to load {graph_path}: {e}')
            corrupted_iids.append(IID)
            continue

    if missing_iids:
        print(f"\nSkipped {len(missing_iids)} IIDs due to not found these file.")
    if corrupted_iids:
        print(f"\nSkipped {len(corrupted_iids)} IIDs due to data errors.")
        print(corrupted_iids[:10])
        if len(corrupted_iids) > 10:
            print(f"... and {len(corrupted_iids) - 10} more")

        skipped_file = (
            f"./logs/skipped_iids_{args.dataset}_{args.atlas}_"
            f"{args.edge_dir_prefix.replace('/', '_')}.txt"
        )
        os.makedirs("./logs", exist_ok=True)
        with open(skipped_file, "w") as f:
            f.write(f"Skipped IIDs for {args.dataset}_{args.atlas}_{args.edge_dir_prefix}\n")
            f.write(f"Missing files: {len(missing_iids)}\n")
            f.write(f"Total data errors: {len(corrupted_iids)}\n")
            f.write("=" * 50 + "\n")
            for iid in corrupted_iids:
                f.write(f"{iid}\n")
        print(f"[INFO] Skipped IIDs saved to: {skipped_file}")

    print(f"[INFO] Successfully loaded {len(dataset)} samples")
    return dataset, labels


def load_data_with_fold_split_combine(args, fold_info):
    """
    Load data and split into train/val/test based on fold information
    
    Args:
        args: Arguments object
        fold_info: Dictionary with 'train_folds', 'val_fold', 'test_fold' keys
    
    Returns:
        tuple: (train_data, val_data, test_data, train_labels, val_labels, test_labels)
    """
    
    # Load fold assignments
    fold_assignments_file = os.path.join(args.folds_dir, '5_folds', 'fold_assignments.csv')
    if not os.path.exists(fold_assignments_file):
        raise FileNotFoundError(f"Fold assignments file not found: {fold_assignments_file}")
    
    fold_df = pd.read_csv(fold_assignments_file, dtype={'IID': str})
    
    # Get IIDs for each fold
    train_folds = fold_info['train_folds']
    val_fold = fold_info['val_fold'] 
    test_fold = fold_info['test_fold']
    
    train_iids = fold_df[fold_df['fold'].isin(train_folds)]['IID'].tolist()
    val_iids = fold_df[fold_df['fold'] == val_fold]['IID'].tolist()
    test_iids = fold_df[fold_df['fold'] == test_fold]['IID'].tolist()
    
    print(f"[INFO] Fold {train_folds} (train): {len(train_iids)} subjects")
    print(f"[INFO] Fold {val_fold} (val): {len(val_iids)} subjects") 
    print(f"[INFO] Fold {test_fold} (test): {len(test_iids)} subjects")
    
    # Load data for each split
    train_data, train_labels = load_data_for_iids_combine(args, train_iids)
    val_data, val_labels = load_data_for_iids_combine(args, val_iids)
    test_data, test_labels = load_data_for_iids_combine(args, test_iids)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels

def load_data_for_iids_combine(args, iids):
    dataset = []
    labels = []
    corrupted_iids = []
    missing_iids = []

    graph_dir = os.path.join(
        args.dataset_dir,
        'graphconstructionedge',
        args.graph_type,
        args.dataset,
        args.mode,
        args.atlas,
        args.edge_dir_prefix
    )

    fold_assignments_file = os.path.join(args.folds_dir, '5_folds', 'fold_assignments.csv')
    fold_df = pd.read_csv(fold_assignments_file, dtype={'IID': str})

    for IID in iids:
        graph_filename = (
            f'dataset-{args.dataset}_sub-{IID}_task-rest_desc-staticgraphconstructionedge_'
            f'atlas-{args.atlas}_contrmethd-{args.edge_dir_prefix}.pkl'
        )
        graph_path = os.path.join(graph_dir, graph_filename)

        if not os.path.exists(graph_path):
            missing_iids.append(IID)
            continue

        try:
            data_batch = torch.load(graph_path, map_location='cpu', weights_only=False)
            data_obj = data_batch[0]

            label_row = fold_df[fold_df['IID'] == IID]
            if len(label_row) == 0:
                corrupted_iids.append(IID)
                continue

            y_val = int(label_row['Diagnosis'].iloc[0])
            if args.dataset in ['ADNI', 'HCP', 'ADHD']:
                y_val = 0 if y_val == 0 else 1
            y = torch.tensor(y_val, dtype=torch.long)

            x = data_obj.x.float()
            edge_index = data_obj.edge_index.long()
            edge_attr = data_obj.edge_attr
            hyper_edge_index = data_obj.hyper_edge_index.long()
            hyper_edge_attr = data_obj.hyper_edge_attr
            if edge_attr is not None:
                edge_attr = edge_attr.float()

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                hyper_edge_index=hyper_edge_index,
                hyper_edge_attr=hyper_edge_attr,
                y=y
            )

            dataset.append(data)
            labels.append(y)

        except Exception as e:
            print(f'[ERROR] Failed to load {graph_path}: {e}')
            corrupted_iids.append(IID)
            continue

    if missing_iids:
        print(f"\nSkipped {len(missing_iids)} IIDs due to not found these file.")
    if corrupted_iids:
        print(f"\nSkipped {len(corrupted_iids)} IIDs due to data errors.")
        print(corrupted_iids[:10])
        if len(corrupted_iids) > 10:
            print(f"... and {len(corrupted_iids) - 10} more")

        skipped_file = (
            f"./logs/skipped_iids_{args.dataset}_{args.atlas}_"
            f"{args.edge_dir_prefix.replace('/', '_')}.txt"
        )
        os.makedirs("./logs", exist_ok=True)
        with open(skipped_file, "w") as f:
            f.write(f"Skipped IIDs for {args.dataset}_{args.atlas}_{args.edge_dir_prefix}\n")
            f.write(f"Missing files: {len(missing_iids)}\n")
            f.write(f"Total data errors: {len(corrupted_iids)}\n")
            f.write("=" * 50 + "\n")
            for iid in corrupted_iids:
                f.write(f"{iid}\n")
        print(f"[INFO] Skipped IIDs saved to: {skipped_file}")

    print(f"[INFO] Successfully loaded {len(dataset)} samples")
    return dataset, labels


def load_data_with_fold_split_ts(args, fold_info):
    """
    Load data and split into train/val/test based on fold information
    
    Args:
        args: Arguments object
        fold_info: Dictionary with 'train_folds', 'val_fold', 'test_fold' keys
    
    Returns:
        tuple: (train_data, val_data, test_data, train_labels, val_labels, test_labels)
    """
    
    # Load fold assignments
    fold_assignments_file = os.path.join(args.folds_dir, '5_folds', 'fold_assignments.csv')
    if not os.path.exists(fold_assignments_file):
        raise FileNotFoundError(f"Fold assignments file not found: {fold_assignments_file}")
    
    fold_df = pd.read_csv(fold_assignments_file, dtype={'IID': str})
    
    # Get IIDs for each fold
    train_folds = fold_info['train_folds']
    val_fold = fold_info['val_fold'] 
    test_fold = fold_info['test_fold']
    
    train_iids = fold_df[fold_df['fold'].isin(train_folds)]['IID'].tolist()
    val_iids = fold_df[fold_df['fold'] == val_fold]['IID'].tolist()
    test_iids = fold_df[fold_df['fold'] == test_fold]['IID'].tolist()
    
    print(f"[INFO] Fold {train_folds} (train): {len(train_iids)} subjects")
    print(f"[INFO] Fold {val_fold} (val): {len(val_iids)} subjects") 
    print(f"[INFO] Fold {test_fold} (test): {len(test_iids)} subjects")
    
    # Load data for each split
    train_data, train_labels = load_data_for_iids_ts(args, train_iids)
    val_data, val_labels = load_data_for_iids_ts(args, val_iids)
    test_data, test_labels = load_data_for_iids_ts(args, test_iids)
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

def load_data_for_iids_ts(args, iids):
    dataset = []
    labels = []
    corrupted_iids = []
    missing_iids = []

    graph_dir = os.path.join(
        args.dataset_dir,
        'graphconstructionedge',
        args.graph_type,
        args.dataset,
        args.mode,
        args.atlas,
        args.edge_dir_prefix
    )

    fold_assignments_file = os.path.join(args.folds_dir, '5_folds', 'fold_assignments.csv')
    fold_df = pd.read_csv(fold_assignments_file, dtype={'IID': str})

    for IID in iids:
        graph_filename = (
            f'dataset-{args.dataset}_sub-{IID}_task-rest_desc-staticgraphconstructionedge_'
            f'atlas-{args.atlas}_contrmethd-{args.edge_dir_prefix}.pkl'
        )
        graph_path = os.path.join(graph_dir, graph_filename)

        if not os.path.exists(graph_path):
            missing_iids.append(IID)
            continue

        try:
            data_batch = torch.load(graph_path, map_location='cpu', weights_only=False)
            data_obj = data_batch[0]

            label_row = fold_df[fold_df['IID'] == IID]
            if len(label_row) == 0:
                corrupted_iids.append(IID)
                continue

            y_val = int(label_row['Diagnosis'].iloc[0])
            if args.dataset in ['ADNI', 'HCP', 'ADHD']:
                y_val = 0 if y_val == 0 else 1
            y = torch.tensor(y_val, dtype=torch.long)

            x = data_obj.x.float()
            corr = data_obj.corr.float()
            edge_index = data_obj.edge_index.long()
            edge_attr = data_obj.edge_attr
            if edge_attr is not None:
                edge_attr = edge_attr.float()

            data = Data(
                x=x,
                corr=corr,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y
            )

            dataset.append(data)
            labels.append(y)

        except Exception as e:
            print(f'[ERROR] Failed to load {graph_path}: {e}')
            corrupted_iids.append(IID)
            continue

    if missing_iids:
        print(f"\nSkipped {len(missing_iids)} IIDs due to not found these file.")
    if corrupted_iids:
        print(f"\nSkipped {len(corrupted_iids)} IIDs due to data errors.")
        print(corrupted_iids[:10])
        if len(corrupted_iids) > 10:
            print(f"... and {len(corrupted_iids) - 10} more")

        skipped_file = (
            f"./logs/skipped_iids_{args.dataset}_{args.atlas}_"
            f"{args.edge_dir_prefix.replace('/', '_')}.txt"
        )
        os.makedirs("./logs", exist_ok=True)
        with open(skipped_file, "w") as f:
            f.write(f"Skipped IIDs for {args.dataset}_{args.atlas}_{args.edge_dir_prefix}\n")
            f.write(f"Missing files: {len(missing_iids)}\n")
            f.write(f"Total data errors: {len(corrupted_iids)}\n")
            f.write("=" * 50 + "\n")
            for iid in corrupted_iids:
                f.write(f"{iid}\n")
        print(f"[INFO] Skipped IIDs saved to: {skipped_file}")

    print(f"[INFO] Successfully loaded {len(dataset)} samples")
    return dataset, labels