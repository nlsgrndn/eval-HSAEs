
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from path_hub import PathBuilder
import os


def _values_equal(a, b):
    """Equality check that handles NaN, lists, and numpy arrays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a) == list(b)
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b

def load_graph_level_metrics(configs, graph_level_metrics) -> pd.DataFrame:
    """Loads the results DataFrame from a pickle file."""
    result_df = None
    for config in configs:
        metric_df = None
        for metric in graph_level_metrics:
            df = metric.load_results(config)
            
            # Skip empty dataframes or dataframes missing required columns
            if df.empty:
                print(f"Warning: Skipping empty dataframe from {metric.__name__} for config {config.simple_name}")
                continue
            
            required_cols = ['config_name']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Skipping dataframe from {metric.__name__} for config {config.simple_name} - missing required columns")
                continue
            
            # join dataframes on common columns
            if metric_df is None:
                metric_df = df
            else:
                # get common columns
                # common_cols = metric_df.columns.intersection(df.columns).tolist()
                common_cols = ['config_name']
                metric_df = pd.merge(metric_df, df, on=common_cols, how='outer')
                assert len(metric_df) == len(df)  # ensure no duplication of rows
        
        # Skip config if no valid metric dataframes were found
        if metric_df is None:
            print(f"Warning: Skipping config {config.simple_name} - no valid metric data found")
            continue
        
        # concatenate dataframes from different configs, ensure that all columns are the same
        if result_df is None:
            result_df = metric_df
        else:
            result_df = pd.concat([result_df, metric_df], ignore_index=True, sort=False)
    
    return result_df

def load_metrics(configs, metrics) -> pd.DataFrame:
    """Loads the results DataFrame from a pickle file."""
    result_df = None
    for config in configs:
        metric_df = None
        for metric in metrics:
            df = metric.load_results(config)
            
            # Skip empty dataframes or dataframes missing required columns
            if df.empty:
                print(f"Warning: Skipping empty dataframe from {metric.__name__} for config {config.simple_name}")
                continue
            
            required_cols = ['config_name', 'node_id', 'parent_sae_id', 'num_children']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Skipping dataframe from {metric.__name__} for config {config.simple_name} - missing required columns")
                continue
            
            # join dataframes on common columns
            if metric_df is None:
                metric_df = df
            else:
                common_cols = ['config_name', 'node_id', 'parent_sae_id', 'num_children']
                # Non-key columns shared between both frames (e.g. 'child_sae_ids', which every
                # cluster runner emits as metadata) must hold equivalent values for the same row;
                # verify and then drop from the incoming frame to avoid suffix collisions.
                duplicate_cols = [c for c in df.columns if c in metric_df.columns and c not in common_cols]
                if duplicate_cols:
                    check = pd.merge(
                        metric_df[common_cols + duplicate_cols],
                        df[common_cols + duplicate_cols],
                        on=common_cols, how='inner', suffixes=('__l', '__r'),
                    )
                    for col in duplicate_cols:
                        left, right = check[f'{col}__l'], check[f'{col}__r']
                        mismatches = sum(1 for l, r in zip(left, right) if not _values_equal(l, r))
                        if mismatches:
                            raise ValueError(
                                f"Column '{col}' has {mismatches} mismatched values between "
                                f"metric runners for the same {common_cols} — cannot merge safely."
                            )
                metric_df = pd.merge(metric_df, df.drop(columns=duplicate_cols), on=common_cols, how='outer')
                assert len(metric_df) == len(df)  # ensure no duplication of rows
        
        # Skip config if no valid metric dataframes were found
        if metric_df is None:
            print(f"Warning: Skipping config {config.simple_name} - no valid metric data found")
            continue
        
        # concatenate dataframes from different configs, ensure that all columns are the same
        if result_df is None:
            result_df = metric_df
        else:
            result_df = pd.concat([result_df, metric_df], ignore_index=True, sort=False)
    
    return result_df


def load_node_metrics(configs, metrics) -> pd.DataFrame:
    """
    Loads per-node metric results from a pickle file.
    
    Unlike load_metrics which expects cluster-level data (with parent_sae_id, num_children),
    this function handles node-level data (just individual node metrics).
    
    Args:
        configs: List of config objects
        metrics: List of metric runner classes
    
    Returns:
        DataFrame with per-node metrics
    """
    result_df = None
    for config in configs:
        metric_df = None
        for metric in metrics:
            df = metric.load_results(config)
            
            # Skip empty dataframes
            if df.empty:
                print(f"Warning: Skipping empty dataframe from {metric.__name__} for config {config.simple_name}")
                continue
            
            # For node-level metrics, we expect simpler columns
            required_cols = ['config_name', 'node']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Skipping dataframe from {metric.__name__} for config {config.simple_name} - missing required columns {required_cols}")
                continue
            
            # Join dataframes on common columns
            if metric_df is None:
                metric_df = df
            else:
                common_cols = ['config_name', 'node', 'subset_sae_id', 'original_sae_id']
                # Only use columns that exist in both
                common_cols = [col for col in common_cols if col in metric_df.columns and col in df.columns]
                metric_df = pd.merge(metric_df, df, on=common_cols, how='outer')
                assert len(metric_df) == len(df), f"Row count mismatch after merge: {len(metric_df)} vs {len(df)}"
        
        # Skip config if no valid metric dataframes were found
        if metric_df is None:
            print(f"Warning: Skipping config {config.simple_name} - no valid metric data found")
            continue
        
        # Concatenate dataframes from different configs
        if result_df is None:
            result_df = metric_df
        else:
            result_df = pd.concat([result_df, metric_df], ignore_index=True, sort=False)
    
    return result_df


def group_by(df, column_name: str = "config_name") -> dict:
    # group the dataframe by dataset and model by config_name
    dfs_dict_by_config = {}
    for config_name in df[column_name].unique():
        subset = df[df[column_name] == config_name].copy()
        dfs_dict_by_config[config_name] = subset
    return dfs_dict_by_config


# TODO: add function to plot distributions in sample plot with same bins, both normalized and absolute frequencies
def plot_distribution(data, title, xlabel, log_scale_y=False, y_lim=None, x_lim=None):
    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=30, kde=True)
    
    # Add mean and median as vertical lines
    mean_val = np.nanmean(data)
    median_val = np.nanmedian(data)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.3f}')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    if log_scale_y:
        plt.yscale('log')
    plt.show()
    plt.close()

def plot_correlation(x, y, title, xlabel, ylabel, hue=None):
    plt.figure(figsize=(8, 6))
    
    # Create scatter plot with optional color coding
    if hue is not None:
        sns.scatterplot(x=x, y=y, hue=hue, alpha=0.6)
    else:
        sns.scatterplot(x=x, y=y, alpha=0.6)
    
    # Calculate and display correlation coefficient
    if len(x) > 1 and len(y) > 1:
        corr_coef = np.corrcoef(x, y)[0, 1]
        
        # Add trend line using numpy polyfit
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
        
        plt.title(f"{title} (Correlation: {corr_coef:.3f})")
    else:
        plt.title(title)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    plt.close()

def plot_confounder(metric, potential_confounder, dfs_dict_by_config):
    # Combine all dataframes and add config_name for coloring
    combined_df = pd.concat([df for config, df in dfs_dict_by_config.items()])
    plot_correlation(
        combined_df[f"{metric}"],
        combined_df[potential_confounder],
        f"Correlation between {metric} and {potential_confounder}",
        f"{metric}",
        f"{potential_confounder}",
        hue=combined_df["config_name"]
    )

def plot_potential_confounders(dfs_dict_by_config, metric, potential_confounders = ["num_children", "num_parent_active"], ):
    for potential_confounder in potential_confounders:
        plot_confounder(metric, potential_confounder, dfs_dict_by_config)
        plot_confounder(metric, potential_confounder, {key: value for key, value in dfs_dict_by_config.items() if key != "cc3m_vit_batchtopk"})

def dummy_function(dfs_dict_by_config, metric_name, plot=False):
    # check whether the metric is single value, 1d array or 2d array
    sample_value = dfs_dict_by_config[list(dfs_dict_by_config.keys())[0]][metric_name].iloc[0]
    if np.isscalar(sample_value):
        print(f"{metric_name} (single value)")
        dummy_single_value(dfs_dict_by_config, metric_name, plot)
    elif isinstance(sample_value, np.ndarray) and sample_value.ndim == 1:
        print(f"{metric_name} (1d array)")
        dummy_1d_arr(dfs_dict_by_config, metric_name, plot)
    elif isinstance(sample_value, np.ndarray) and sample_value.ndim == 2:
        print(f"{metric_name} (2d array)")
        dummy_2d_arr(dfs_dict_by_config, metric_name, plot)
    else:
        raise ValueError(f"Unsupported metric type for {metric_name}")

def dummy_single_value(dfs_dict_by_config, metric_name, plot=False):
    for config_name, df in dfs_dict_by_config.items():
        mean_of_metric = df[metric_name].mean()
        std_of_metric = df[metric_name].std()
        print(f"{config_name}: Mean  = {mean_of_metric:.4f}, Std  = {std_of_metric:.4f}")
    if plot:
        for config_name, df in dfs_dict_by_config.items():
            plot_distribution(df[metric_name], f"{metric_name} Distribution for {config_name}", f"{metric_name} Distribution")

def dummy_1d_arr(dfs_dict_by_config, metric_name, plot=False):
    for config_name, df in dfs_dict_by_config.items():
        metric_values_collector =[]
        count_nans = 0
        full_length = 0
        for _, row in df.iterrows():
            metric_values_collector.append(row[metric_name])
            count_nans += np.isnan(row[metric_name]).sum()
            full_length += len(row[metric_name])
        metric_values_collector = np.concatenate(metric_values_collector)
        mean_of_metric = np.nanmean(metric_values_collector)
        std_of_metric = np.nanstd(metric_values_collector)
        print(f"{config_name}: Mean  = {mean_of_metric:.4f}, Std  = {std_of_metric:.4f}, NaNs = {count_nans}, Total = {full_length}")
        if plot:
            plot_distribution(metric_values_collector, f"{metric_name} Distribution for {config_name}", f"{metric_name} Distribution")

def dummy_2d_arr(dfs_dict_by_config, metric_name, plot=False):
    for config_name, df in dfs_dict_by_config.items():
        total_cols = 0
        nan_cols = 0
        all_values = []
        mean_values = []
        for matrix in df[metric_name]:
            # get upper triangle excluding diagonal
            upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
            
            all_values.extend(upper_triangle[~np.isnan(upper_triangle)])
            
            mean_values.append(np.nanmean(upper_triangle))

            total_cols += matrix.shape[1]
            number_of_cols_with_all_nan = np.sum(np.all(np.isnan(matrix), axis=0))
            nan_cols += number_of_cols_with_all_nan
        df.loc[:, f"{metric_name}_mean"] = mean_values
        mean_of_means = np.nanmean(mean_values)
        std_of_means = np.nanstd(mean_values)
        print(f"{config_name} - {metric_name}: Total cols = {total_cols}, Cols with all NaN = {nan_cols}, Proportion = {nan_cols / total_cols:.4f}")
        print(f"{config_name} - {metric_name}: Mean = {mean_of_means:.4f}, Std = {std_of_means:.4f}")
        if plot:
            plot_distribution(all_values, f"{metric_name} Distribution for {config_name}", metric_name, log_scale_y=False, y_lim=(0.1, None), x_lim = (0,1) if metric_name == "jaccard_similarity_matrix" else None)


# DataFrame-returning versions for script usage
def compute_single_value_metrics(dfs_dict_by_config, metric_name):
    """Compute statistics for single-value metrics and return as DataFrame."""
    results = []
    for config_name, df in dfs_dict_by_config.items():
        values = df[metric_name]
        mean_val = values.mean()
        std_val = values.std()
        nan_count = values.isna().sum()
        total_count = len(values)
        results.append({
            'config_name': config_name,
            'metric_name': metric_name,
            'mean': mean_val,
            'std': std_val,
            'total_count': total_count,
            'nan_count': nan_count,
            'nan_proportion': nan_count / total_count if total_count > 0 else 0.0
        })
    return pd.DataFrame(results)


def compute_1d_array_metrics(dfs_dict_by_config, metric_name):
    """Compute statistics for 1D array metrics and return as DataFrame."""
    results = []
    for config_name, df in dfs_dict_by_config.items():
        metric_values_collector = []
        count_nans = 0
        full_length = 0
        for _, row in df.iterrows():
            metric_values_collector.append(row[metric_name])
            count_nans += np.isnan(row[metric_name]).sum()
            full_length += len(row[metric_name])
        metric_values_collector = np.concatenate(metric_values_collector)
        mean_val = np.nanmean(metric_values_collector)
        std_val = np.nanstd(metric_values_collector)
        results.append({
            'config_name': config_name,
            'metric_name': metric_name,
            'mean': mean_val,
            'std': std_val,
            'total_values': full_length,
            'nan_count': count_nans,
            'nan_proportion': count_nans / full_length if full_length > 0 else 0.0
        })
    return pd.DataFrame(results)


def compute_2d_array_metrics(dfs_dict_by_config, metric_name):
    """Compute statistics for 2D array metrics and return as DataFrame."""
    results = []
    for config_name, df in dfs_dict_by_config.items():
        total_cols = 0
        nan_cols = 0
        mean_values = []
        for matrix in df[metric_name]:
            # get upper triangle excluding diagonal
            upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
            mean_values.append(np.nanmean(upper_triangle))
            total_cols += matrix.shape[1]
            number_of_cols_with_all_nan = np.sum(np.all(np.isnan(matrix), axis=0))
            nan_cols += number_of_cols_with_all_nan
        
        mean_of_means = np.nanmean(mean_values)
        std_of_means = np.nanstd(mean_values)
        nan_proportion = nan_cols / total_cols if total_cols > 0 else 0.0
        
        results.append({
            'config_name': config_name,
            'metric_name': metric_name,
            'mean': mean_of_means,
            'std': std_of_means,
            'total_cols': total_cols,
            'nan_cols': nan_cols,
            'nan_proportion': nan_proportion
        })
    return pd.DataFrame(results)


