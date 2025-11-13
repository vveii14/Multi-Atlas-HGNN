import pdb
import torch
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.linalg import pinv
from lingam import DirectLiNGAM
import matplotlib.pyplot as plt
from scipy.signal import coherence
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from torch_geometric.data import Data, Batch
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_sklearn
import scipy

def Eu_dis(x, metric='euclidean'):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    d = scipy.spatial.distance.pdist(x, metric)
    # 用于计算样本对之间的欧式距离
    d = scipy.spatial.distance.squareform(d)
    # 将样本间距离用方阵表示出来
    return d

def construct_H_with_KNN_from_distance(feature, k_neig, is_probH = True, normalize = False):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    Use the mean of the input feature vectors to compute distances, but keep the original
    feature matrix as node attributes `x` in the returned Data object.

    Returns a torch_geometric.data.Data with attributes x and H (torch tensors).
    """

    dis_mat = Eu_dis(feature.T)

    n_obj = dis_mat.shape[0]

    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))

    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx, :]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (2 * avg_dis ** 2))
            else:
                H[node_idx, center_idx] = 1.0

    if normalize:
        for i in range(np.size(H,1)):
            sum = np.sum(H[:, i])
            H[:, i] = H[:, i]/sum

    # Build torch_geometric Data with only x and H
    x = torch.tensor(dis_mat, dtype=torch.float)
    data = Data(x=x)
    data.H = torch.tensor(H, dtype=torch.float)
    return data

def correlation_matrix_to_hypergraph_data(corr_matrix, k_neig=10, is_probH=True):

    if isinstance(corr_matrix, torch.Tensor):
        corr = corr_matrix.detach().cpu().numpy()
    else:
        corr = np.array(corr_matrix, copy=False)

    R = corr.shape[0]
    k = int(k_neig)              
    assert 0 < k <= R, "k_neig 必须在 1~R 之间"

    H = np.zeros((R, R), dtype=float)

    for center in range(R):
        row = corr[center].copy()
        row[center] = 1.0

        idx = np.argsort(-np.abs(row))

        topk = idx[:k].copy()
        if center not in topk:
            topk[-1] = center

        for node in topk:
            val = np.abs(row[node]) if is_probH else 1.0
            if val > H[node, center]:
                H[node, center] = val

    rows, cols = np.nonzero(H)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)   
    edge_attr = torch.tensor(H[rows, cols], dtype=torch.float32)

    x = torch.tensor(corr, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.H = torch.tensor(H, dtype=torch.float32)

    return data

def timeseries_correlation_matrix_to_graph_data(corr_matrix, ts, k_neig=10, is_probH=True, T_keep=76):

    if isinstance(corr_matrix, torch.Tensor):
        corr = corr_matrix.detach().cpu().numpy()
    else:
        corr = np.array(corr_matrix, copy=False)

    R = corr.shape[0]
    k = int(k_neig)              
    assert 0 < k <= R, "k_neig 必须在 1~R 之间"

    H = np.zeros((R, R), dtype=float)

    for center in range(R):
        row = corr[center].copy()
        row[center] = 1.0

        idx = np.argsort(-np.abs(row))

        topk = idx[:k].copy()
        if center not in topk:
            topk[-1] = center

        for node in topk:
            val = np.abs(row[node]) if is_probH else 1.0
            if val > H[node, center]:
                H[node, center] = val

    if isinstance(ts, torch.Tensor):
        ts_np = ts.detach().cpu().numpy()
    else:
        ts_np = np.array(ts, copy=False)

    ts_min = ts_np.min()
    ts_max = ts_np.max()
    ts_norm = 2 * (ts_np - ts_min) / (ts_max - ts_min + 1e-8) - 1

    ts_norm = ts_norm[:T_keep, :]
    assert ts_norm.shape[1] == corr.shape[0]

    ts_norm = ts_norm.T           
    x = torch.tensor(ts_norm, dtype=torch.float32)

    rows, cols = np.nonzero(H)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_attr = torch.tensor(H[rows, cols], dtype=torch.float32)
    corr = torch.tensor(corr, dtype=torch.float32)

    data = Data(x=x, corr=corr, edge_index=edge_index, edge_attr=edge_attr)
    data.H = torch.tensor(H, dtype=torch.float32)

    return data

def timeseries_roi_correlation_matrix_to_graph_data(corr_matrix, ts, k_neig=10, is_probH=True, T_keep=76):

    if isinstance(corr_matrix, torch.Tensor):
        corr = corr_matrix.detach().cpu().numpy()
    else:
        corr = np.array(corr_matrix, copy=False)

    R = corr.shape[0]
    k = int(k_neig)              
    assert 0 < k <= R, "k_neig 必须在 1~R 之间"

    H = np.zeros((R, R), dtype=float)

    for center in range(R):
        row = corr[center].copy()
        row[center] = 1.0

        idx = np.argsort(-np.abs(row))

        topk = idx[:k].copy()
        if center not in topk:
            topk[-1] = center

        for node in topk:
            val = np.abs(row[node]) if is_probH else 1.0
            if val > H[node, center]:
                H[node, center] = val

    if isinstance(ts, torch.Tensor):
        ts_np = ts.detach().cpu().numpy()
    else:
        ts_np = np.array(ts, copy=False)

    ts_min = ts_np.min()
    ts_max = ts_np.max()
    ts_norm = 2 * (ts_np - ts_min) / (ts_max - ts_min + 1e-8) - 1

    ts_norm = ts_norm[:T_keep, :]
    assert ts_norm.shape[1] == corr.shape[0]

    ts_norm = ts_norm.T           
    x = torch.tensor(ts_norm, dtype=torch.float32)
    x_corr = torch.tensor(corr, dtype=torch.float32)
    x = torch.cat((x, x_corr), dim=1)

    rows, cols = np.nonzero(H)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_attr = torch.tensor(H[rows, cols], dtype=torch.float32)
    corr = torch.tensor(corr, dtype=torch.float32)

    data = Data(x=x, corr=corr, edge_index=edge_index, edge_attr=edge_attr)
    data.H = torch.tensor(H, dtype=torch.float32)

    return data


def correlation_matrix_to_graph_data(corr_matrix):
    num_nodes = corr_matrix.shape[0]

    edge_index = np.nonzero(corr_matrix)  
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_attr = torch.tensor(corr_matrix[edge_index[0], edge_index[1]], dtype=torch.float)

    x = torch.tensor(corr_matrix, dtype=torch.float) 

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def combine_correlation_matrix_to_hypergraph_data(corr_matrix, k_neig=10, is_probH=True):

    if isinstance(corr_matrix, torch.Tensor):
        corr = corr_matrix.detach().cpu().numpy()
    else:
        corr = np.array(corr_matrix, copy=False)

    R = corr.shape[0]
    k = int(k_neig)              
    assert 0 < k <= R, "k_neig 必须在 1~R 之间"

    H = np.zeros((R, R), dtype=float)

    for center in range(R):
        row = corr[center].copy()
        row[center] = 1.0

        idx = np.argsort(-np.abs(row))

        topk = idx[:k].copy()
        if center not in topk:
            topk[-1] = center

        for node in topk:
            val = np.abs(row[node]) if is_probH else 1.0
            if val > H[node, center]:
                H[node, center] = val

    rows, cols = np.nonzero(H)
    hyper_edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)   
    hyper_edge_attr = torch.tensor(H[rows, cols], dtype=torch.float32)

    x = torch.tensor(corr, dtype=torch.float32)
    edge_index = np.nonzero(corr_matrix)  
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(corr_matrix[edge_index[0], edge_index[1]], dtype=torch.float32)

    data = Data(x=x, hyper_edge_index=hyper_edge_index, hyper_edge_attr=hyper_edge_attr, edge_index=edge_index, edge_attr=edge_attr)
    data.H = torch.tensor(H, dtype=torch.float32)

    return data


def pearson_correlation(data):
    return np.corrcoef(data, rowvar=False)

def cosine_similarity(data):
    return cosine_similarity_sklearn(data.T)
    
def partial_correlation(data):
    # Compute the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    
    # Compute the precision matrix (inverse of the covariance matrix)
    precision_matrix = pinv(cov_matrix)
    
    # Compute partial correlations from the precision matrix
    diag = np.diag(precision_matrix)
    partial_corr_matrix = -precision_matrix / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(partial_corr_matrix, 1.0)  # Set diagonal to 1.0
    
    return partial_corr_matrix

def correlations_correlation(data):
    pearson_correlation_matrix = np.corrcoef(data, rowvar=False)
    n = pearson_correlation_matrix.shape[0]
    high_order_fc_matrix = np.zeros((n, n))

    # Step 2: Compute the correlation's correlation for each pair of ROIs
    for row_index in range(n):
        for col_index in range(row_index + 1, n):  # Only need to compute upper triangular part
            high_order_fc_matrix[row_index, col_index] = np.corrcoef(pearson_correlation_matrix[row_index, :], pearson_correlation_matrix[col_index, :])[0, 1]
            high_order_fc_matrix[col_index, row_index] = high_order_fc_matrix[row_index, col_index]  # Symmetric matrix
    return high_order_fc_matrix

def associated_high_order_fc(data):
    pearson_correlation_matrix = np.corrcoef(data, rowvar=False)

    # Calculate the Correlation's Correlation (High-order FC)
    n = pearson_correlation_matrix.shape[0]
    high_order_fc_matrix = np.zeros((n, n))

    for row_index in range(n):
        for col_index in range(row_index + 1, n):  # Only need to compute upper triangular part
            high_order_fc_matrix[row_index, col_index] = np.corrcoef(pearson_correlation_matrix[row_index, :], pearson_correlation_matrix[col_index, :])[0, 1]
            high_order_fc_matrix[col_index, row_index] = high_order_fc_matrix[row_index, col_index]  # Symmetric matrix

    # Calculate the Associated High-order Functional Connectivity Matrix
    associated_high_order_fc_matrix = np.zeros((n, n))

    for row_index in range(n):
        for col_index in range(row_index + 1, n):
            associated_high_order_fc_matrix[row_index, col_index] = np.corrcoef(pearson_correlation_matrix[row_index, :], high_order_fc_matrix[col_index, :])[0, 1]
            associated_high_order_fc_matrix[col_index, row_index] = associated_high_order_fc_matrix[row_index, col_index]  # Symmetric matrix

    return associated_high_order_fc_matrix

def euclidean_distance(data):
    n_vars = data.shape[1]
    # calculate the euclidean distance matrix
    euclidean_distance_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            euclidean_distance_matrix[i, j] = np.linalg.norm(data.iloc[:, i] - data.iloc[:, j])
            euclidean_distance_matrix[j, i] = euclidean_distance_matrix[i, j]
    return euclidean_distance_matrix

def knn_graph(data, K = 20):

    # Number of nearest neighbors
    K = 20

    # Calculate the KNN graph, using  weighted by the Gaussian similarity function of Euclidean distances,

    knn_graph = kneighbors_graph(data.T, K, mode='distance', metric='euclidean', include_self=False)

    # knn_graph to adjacency matrix

    knn_graph_adj_matrix = np.zeros((data.shape[1], data.shape[1]))

    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if i != j:
                knn_graph_adj_matrix[i, j] = knn_graph[i, j]

    return knn_graph_adj_matrix

def spearman_correlation(data):
    matrix = data.corr(method='spearman')

    spearman_correlation_matrix = np.zeros((data.shape[1], data.shape[1]))

    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            spearman_correlation_matrix[i, j] = matrix.iloc[i, j]


    return spearman_correlation_matrix

def kendall_correlation(data):
    matrix = data.corr(method='kendall')

    kendall_correlation_matrix = np.zeros((data.shape[1], data.shape[1]))

    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            kendall_correlation_matrix[i, j] = matrix.iloc[i, j]


    return kendall_correlation_matrix

def mutual_information(data):
    num_rois = data.shape[1]
    mutual_info_matrix = np.zeros((num_rois, num_rois))
    
    for i in range(num_rois):
        for j in range(num_rois):
            if i == j:
                mutual_info_matrix[i, j] = 0  # Mutual information with itself is not needed
            else:
                # Estimate mutual information between the two time series
                mutual_info_matrix[i, j] = mutual_info_regression(data.iloc[:, i].values.reshape(-1, 1), 
                                                                  data.iloc[:, j].values)[0]

    return mutual_info_matrix

def cross_correlation(data):
    num_rois = data.shape[1]
    cross_corr_matrix = np.zeros((num_rois, num_rois))

    for i in range(num_rois):    
        for j in range(num_rois):
            cross_corr_matrix[i, j] = np.corrcoef(data.iloc[:, i], data.iloc[:, j])[0, 1]
    return cross_corr_matrix

def granger_causality(data, max_lag=1):
    num_rois = data.shape[1]
    granger_matrix = np.zeros((num_rois, num_rois))

    for i in range(num_rois):
        for j in range(num_rois):
            if i != j:
                test_result = grangercausalitytests(data[[data.columns[i], data.columns[j]]], max_lag, verbose=False)
                p_values = [round(test[0]['ssr_ftest'][1], 4) for test in test_result.values()]
                granger_matrix[i, j] = np.min(p_values)
    
    return granger_matrix

def coherence_matrix(data, fs=1.0, nperseg=None):
    """
    Calculate the coherence matrix for a set of time series data.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): A 2D array where rows are time points and columns are different ROI time series.
    fs (float, optional): The sampling frequency of the data. Default is 1.0.
    nperseg (int, optional): Length of each segment for calculating coherence. Default is None, which uses scipy's default.

    Returns:
    numpy.ndarray: A symmetric matrix of coherence values.
    """
    # Convert DataFrame to numpy array if necessary
    if isinstance(data, pd.DataFrame):
        data = data.values

    n = data.shape[1]  # Number of ROIs
    coherence_matrix = np.zeros((n, n))  # Initialize an empty coherence matrix

    nperseg = nperseg or data.shape[0] // 4  # Default to 25% of the data length

    for i in range(n):
        for j in range(i, n):
            # Compute coherence between ROI i and ROI j
            f, Cxy = coherence(data[:, i], data[:, j], fs=fs, nperseg=nperseg)
            coherence_matrix[i, j] = np.mean(Cxy)  # Average coherence over all frequencies
            coherence_matrix[j, i] = coherence_matrix[i, j]  # Symmetric matrix

    return coherence_matrix

def time_delay_embedding(time_series, embedding_dim, time_delay):
    """
    Perform time-delay embedding for a given time series.

    Parameters:
    time_series (numpy.ndarray): The input time series to be embedded.
    embedding_dim (int): The embedding dimension.
    time_delay (int): The time delay.

    Returns:
    numpy.ndarray: The embedded time series in a higher dimensional space.
    """
    N = len(time_series)
    if embedding_dim * time_delay > N:
        raise ValueError(f"Time series too short for the desired embedding parameters. "
                         f"Series length: {N}, required length: {embedding_dim * time_delay}")
    
    # Generate the embedded matrix ensuring each element has the same length
    embedded = []
    for i in range(embedding_dim):
        start_idx = i * time_delay
        end_idx = N - (embedding_dim - 1) * time_delay
        embedded.append(time_series[start_idx:start_idx + end_idx])
    
    embedded = np.array(embedded).T  # Convert list of arrays to 2D numpy array and transpose
    return embedded

def time_delay_embedding(time_series, embedding_dim, time_delay):
    """Perform time-delay embedding on a single time series."""
    n = len(time_series)
    if n < embedding_dim * time_delay:
        raise ValueError("Time series too short for embedding.")
    embedded = np.array([time_series[t : t + embedding_dim * time_delay : time_delay] 
                         for t in range(n - embedding_dim * time_delay + 1)])
    return embedded

def compute_gs_for_pair(i, j, embedded_list):
    """Compute generalised synchronisation for a pair (i, j)."""
    embedded_1 = embedded_list[i]
    embedded_2 = embedded_list[j]
    
    dist_1 = cdist(embedded_1, embedded_1)
    dist_2 = cdist(embedded_2, embedded_2)
    
    nn_1 = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(dist_1).kneighbors(dist_1, return_distance=False)[:, 1]
    
    synchronisation_sum = 0
    for k in range(len(embedded_1)):
        l = nn_1[k]
        synchronisation_sum += dist_2[k, l]
    
    gs_value = synchronisation_sum / len(embedded_1)
    return i, j, gs_value

def generalised_synchronisation_matrix(data, embedding_dim=10, time_delay=1, n_jobs=1):
    """
    Calculate the generalised synchronisation matrix for a set of time series data.

    Parameters:
    - data (pandas.DataFrame or numpy.ndarray): A 2D array where rows are time points and columns are ROI time series.
    - embedding_dim (int, optional): Embedding dimension for time-delay embedding. Default is 10.
    - time_delay (int, optional): Time delay for time-delay embedding. Default is 1.
    - n_jobs (int, optional): Number of parallel jobs. -1 uses all available CPU cores.

    Returns:
    - numpy.ndarray: A symmetric matrix of generalised synchronisation values.
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values

    n = data.shape[1]  # Number of ROIs
    gs_matrix = np.zeros((n, n))  # Initialize the GS matrix

    # Precompute embeddings for all time series
    embedded_list = []
    for i in range(n):
        time_series = data[:, i]
        try:
            embedded = time_delay_embedding(time_series, embedding_dim, time_delay)
            embedded_list.append(embedded)
        except ValueError as e:
            print(f"Skipping ROI {i} due to embedding error: {e}")
            embedded_list.append(None)

    # Generate all valid pairs
    pairs = [(i, j) for i in range(n) for j in range(i, n) 
             if embedded_list[i] is not None and embedded_list[j] is not None]

    # Compute GS values in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(compute_gs_for_pair)(i, j, embedded_list) 
                                      for i, j in pairs)

    # Fill the GS matrix
    for i, j, gs_value in results:
        gs_matrix[i, j] = gs_value
        gs_matrix[j, i] = gs_value

    return gs_matrix

def binarize_data(data, threshold=0.5):
    """
    Binarize the input data based on a threshold.

    Parameters:
    data (numpy.ndarray): Input data with shape (time_points, ROIs).
    threshold (float): Threshold for binarization.

    Returns:
    numpy.ndarray: Binarized data.
    """
    return (data > threshold).astype(int)

def calculate_conditional_probabilities(data):
    """
    Calculate the conditional probabilities for each pair of variables.

    Parameters:
    data (numpy.ndarray): Binarized data with shape (time_points, ROIs).

    Returns:
    tuple: (P(X), P(Y), P(X|Y), P(Y|X), P(X,Y))
    """
    n = data.shape[1]
    P_X = np.mean(data, axis=0)  # P(X) for each ROI
    P_Y = P_X.copy()  # P(Y) is the same as P(X) for each ROI

    P_X_given_Y = np.zeros((n, n))
    P_Y_given_X = np.zeros((n, n))
    P_XY = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                P_X_given_Y[i, j] = 1.0
                P_Y_given_X[i, j] = 1.0
                P_XY[i, j] = P_X[i]
            else:
                # Calculate joint probability P(X,Y)
                P_XY[i, j] = np.mean((data[:, i] == 1) & (data[:, j] == 1))
                # Calculate P(X|Y) = P(X,Y) / P(Y)
                if P_Y[j] != 0:
                    P_X_given_Y[i, j] = P_XY[i, j] / P_Y[j]
                # Calculate P(Y|X) = P(X,Y) / P(X)
                if P_X[i] != 0:
                    P_Y_given_X[i, j] = P_XY[i, j] / P_X[i]

    return P_X, P_Y, P_X_given_Y, P_Y_given_X, P_XY

def patels_conditional_dependence_measures_kappa(data, threshold=0.5):
    """
    Calculate the kappa (κ) matrix of Patel's Conditional Dependence Measures for the input data.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): Input data with shape (time_points, ROIs).
    threshold (float, optional): Threshold for binarization. Default is 0.5.

    Returns:
    numpy.ndarray: Kappa matrix with diagonal elements set to 1.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Step 1: Binarize the data
    binarized_data = binarize_data(data, threshold=threshold)

    # Step 2: Calculate conditional probabilities
    P_X, P_Y, P_X_given_Y, _, _ = calculate_conditional_probabilities(binarized_data)

    n = data.shape[1]
    kappa_matrix = np.zeros((n, n))

    # Step 3: Calculate kappa (κ) matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                kappa_matrix[i, j] = 1.0  # Set diagonal elements to 1
            else:
                # κ = P(X|Y) - P(X)
                kappa_matrix[i, j] = P_X_given_Y[i, j] - P_X[i]

    return kappa_matrix

def patels_conditional_dependence_measures_tau(data, threshold=0.5):
    """
    Calculate the tau (τ) matrix of Patel's Conditional Dependence Measures for the input data.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): Input data with shape (time_points, ROIs).
    threshold (float, optional): Threshold for binarization. Default is 0.5.

    Returns:
    numpy.ndarray: Tau matrix with diagonal elements set to 1.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Step 1: Binarize the data
    binarized_data = binarize_data(data, threshold=threshold)

    # Step 2: Calculate conditional probabilities
    _, _, P_X_given_Y, P_Y_given_X, _ = calculate_conditional_probabilities(binarized_data)

    n = data.shape[1]
    tau_matrix = np.zeros((n, n))

    # Step 3: Calculate tau (τ) matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                tau_matrix[i, j] = 1.0  # Set diagonal elements to 1
            else:
                # τ = P(X|Y) - P(Y|X)
                tau_matrix[i, j] = P_X_given_Y[i, j] - P_Y_given_X[i, j]

    return tau_matrix

def plot_correlation_matrix(correlation_matrix, method=""):
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", xticklabels=True, yticklabels=True)
    # plt.title("Cross-Correlation Matrix")
    plt.title("Correlation Matrix" + " of " + method)
    plt.show()




