import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
        
from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import ClusterActsData, IntraClusterActsMetric


class CoActivationCorrelationMetric(IntraClusterActsMetric):
    """
    Co-activation correlation for each parent-child pair.
    Computes Pearson and Spearman correlation when both parent and child are active.
    Returns 1D arrays with one value per child.
    """
    
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        from scipy.stats import spearmanr
        
        parent_active_mask = cluster_data.parent_binary
        
        num_children = cluster_data.num_children
        coactivation_corr_arr = np.full(num_children, np.nan)
        coactivation_spearman_arr = np.full(num_children, np.nan)
        num_both_active_arr = np.zeros(num_children, dtype=int)
        
        for child_idx in range(num_children):
            child_acts = cluster_data.children_acts[:, child_idx]
            child_binary = cluster_data.children_binary[:, child_idx]
            
            # Restrict to datapoints where both p>THRESHOLD and c>THRESHOLD
            both_active_mask = parent_active_mask & child_binary
            num_both_active_arr[child_idx] = both_active_mask.sum()
            
            if both_active_mask.sum() >= self.MIN_SAMPLES:
                parent_acts_both_active = cluster_data.parent_acts[both_active_mask]
                child_acts_both_active = child_acts[both_active_mask]
                

                if parent_acts_both_active.std() > 0 and child_acts_both_active.std() > 0:
                    # Calculate Pearson correlation
                    coactivation_corr_arr[child_idx] = np.corrcoef(parent_acts_both_active, child_acts_both_active)[0, 1]
                
                    # Calculate Spearman correlation
                    spearman_corr, _ = spearmanr(parent_acts_both_active, child_acts_both_active)
                    coactivation_spearman_arr[child_idx] = spearman_corr

        
        return {
            'coactivation_correlation': coactivation_corr_arr,
            'coactivation_spearman': coactivation_spearman_arr,
            'num_both_active': num_both_active_arr
        }


class NumParentActiveMetric(IntraClusterActsMetric):
    """
    Computes the number of datapoints where parent is active.
    Simple count metric for tracking parent activation frequency.
    """
    
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        parent_active_mask = cluster_data.parent_binary
        
        return {
            'num_parent_active': int(parent_active_mask.sum())
        }

class ChildActivationRatesMetric(IntraClusterActsMetric):
    """
    Computes binary activation rates of children in relation to parent activation.
    
    Returns:
    - child_base_rates: P(c_i > threshold) for each child (1D array)
    - child_rates_over_parent: P(c_i > threshold) / P(p > threshold) for each child (1D array)
    
    Measures how frequently each child activates overall and relative to parent.
    """
    
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        
        # Base activation rates for each child: P(c_i > threshold)
        child_base_rates = cluster_data.children_binary.mean(axis=0)
        
        # Parent activation rate: P(p > threshold)
        parent_base_rate = cluster_data.parent_binary.mean()
        
        # Child rates divided by parent rate
        child_rates_over_parent = child_base_rates / parent_base_rate
        
        return {
            'child_base_rates': child_base_rates,
            'child_rates_over_parent': child_rates_over_parent
        }

class AverageChildrenActiveMetric(IntraClusterActsMetric):
    """
    Computes average number of children active when parent is active.
    Measures how many children typically co-activate.
    """
    
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        parent_active_mask = cluster_data.parent_binary
        
        children_binary_active = cluster_data.children_binary[parent_active_mask]
        num_children_active_per_datapoint = children_binary_active.sum(axis=1)
        avg_num_children_active = num_children_active_per_datapoint.mean()
        # add fraction of children active
        fraction_children_active = num_children_active_per_datapoint / cluster_data.num_children
        avg_fraction_children_active = fraction_children_active.mean()
        
        return {
            'avg_num_children_active': avg_num_children_active,
            'avg_fraction_children_active': avg_fraction_children_active
        }

class CoverageMetric(IntraClusterActsMetric):
    """
    Computes coverage: fraction of parent-active datapoints where at least one child is active.
    P(∃i: c_i > threshold | p > threshold)
    High coverage means children capture most parent activations.
    """
    
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        parent_active_mask = cluster_data.parent_binary
        
        children_binary_active = cluster_data.children_binary[parent_active_mask]
        any_child_active = children_binary_active.any(axis=1)
        coverage = any_child_active.mean()
        
        return {
            'coverage': coverage
        }

class MaskedCosineSimilarityWithParentMetric(IntraClusterActsMetric):
    """
    Computes masked cosine similarity between each child's activations and the parent's activations.
    Masked to only datapoints where child is active.
    Measures alignment of child activations with parent activations.
    """
    
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:

        num_children = cluster_data.num_children
        cosine_sim_arr = np.full(num_children, np.nan)

        for child_idx in range(num_children):
            child_acts = cluster_data.children_acts[:, child_idx]
            child_binary = cluster_data.children_binary[:, child_idx]
            
            # Mask to datapoints where child is active
            if child_binary.sum() >= self.MIN_SAMPLES:
                parent_acts_masked = cluster_data.parent_acts[child_binary].reshape(1, -1)
                child_acts_masked = child_acts[child_binary].reshape(1, -1)
                
                # Compute cosine similarity
                cosine_sim_arr[child_idx] = cosine_similarity(parent_acts_masked, child_acts_masked)[0, 0]

        return {
            'masked_cosine_similarity_with_parent': cosine_sim_arr
        }

class ConditionalActivationMetric(IntraClusterActsMetric):
    """
    Computes P(p > threshold | c_i > threshold) for each child c_i.
    Measures how likely parent is active when each child is active.
    """
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        num_children = cluster_data.num_children
        conditional_activation_arr = np.full(num_children, np.nan)
        
        for child_idx in range(num_children):
            child_binary = cluster_data.children_binary[:, child_idx]
            
            # P(p > threshold | c_i > threshold)
            if child_binary.sum() >= self.MIN_SAMPLES:
                parent_active_given_child = cluster_data.parent_binary[child_binary]
                conditional_activation_arr[child_idx] = parent_active_given_child.mean()
        
        return {
            'conditional_activation': conditional_activation_arr
        }


class ConditionalCorrChildrenOnlyMetric(IntraClusterActsMetric):
    """
    Computes conditional Pearson and Spearman correlation: corr(a_p, S | S > τ_S).
    
    For each datapoint t:
    - a_p,t: parent activation
    - S_t = Σ_i a_{c_i,t}: sum of child activations
    - τ_S: threshold for child sum (uses any child active to define)
    
    Computes Pearson and Spearman correlation between a_p and S on datapoints where children are above threshold.
    Answers: "Among datapoints where children matter, does larger child sum 
    reliably come with larger parent activation?"
    
    Unlike ConditionalCorrMetric, this does NOT require parent to be active.
    High correlation indicates strong relationship when children are active.
    """
    
    def __init__(self, threshold, aggregation_method):
        super().__init__()

        self.threshold = threshold
        self.aggregation_method = aggregation_method
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        from scipy.stats import spearmanr
        
        # Sum of child activations for all datapoints
        if self.aggregation_method == 'sum':
            aggregation = cluster_data.children_acts.sum(axis=1)
        elif self.aggregation_method == 'max':
            aggregation = cluster_data.children_acts.max(axis=1)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
        
        # Condition 1: at least one child is active (no parent requirement)
        any_child_active = cluster_data.children_binary.any(axis=1)
        # Condition 2: aggregation > threshold
        above_threshold = (aggregation > self.threshold)

        selection_condition = any_child_active & above_threshold
        
        # Get activations where condition holds
        a_p_conditioned = cluster_data.parent_acts[selection_condition]
        aggr_conditioned = aggregation[selection_condition]
        
        # Calculate correlations if we have enough samples
        if len(a_p_conditioned) >= self.MIN_SAMPLES:
            # Pearson correlation
            if a_p_conditioned.std() > 0 and aggr_conditioned.std() > 0:
                pearson_corr = np.corrcoef(a_p_conditioned, aggr_conditioned)[0, 1]
            else:
                pearson_corr = np.nan
            
            # Spearman correlation
            try:
                if a_p_conditioned.std() > 0 and aggr_conditioned.std() > 0:
                    spearman_corr, p_value = spearmanr(a_p_conditioned, aggr_conditioned)
                else:
                    spearman_corr = np.nan
                    p_value = np.nan
            except:
                spearman_corr = np.nan
                p_value = np.nan

            if a_p_conditioned.std() > 0 and aggr_conditioned.std() > 0:
                cosine_sim = cosine_similarity(
                    a_p_conditioned.reshape(1, -1),
                    aggr_conditioned.reshape(1, -1)
                )[0, 0]
            else:
                cosine_sim = np.nan

        else:
            pearson_corr = np.nan
            spearman_corr = np.nan
            p_value = np.nan
            cosine_sim = np.nan
        
        return {
            f'conditional_pearson_corr_children_only_{self.aggregation_method}_{self.threshold}': pearson_corr,
            f'conditional_spearman_corr_children_only_{self.aggregation_method}_{self.threshold}': spearman_corr,
            f'conditional_spearman_pvalue_children_only_{self.aggregation_method}_{self.threshold}': p_value,
            f'num_datapoints_conditioned_children_only_{self.aggregation_method}_{self.threshold}': int(selection_condition.sum()),
            f'conditional_cosine_similarity_children_only_{self.aggregation_method}_{self.threshold}': cosine_sim
        }


class ChildParentMagnitudeMetric(IntraClusterActsMetric):
    """
    Computes magnitude behavior of child and parent when both are active.
    
    Returns:
    - fraction_parent_larger_than_child: fraction where a_p > a_c (given both active) (1D array)
    - avg_child_parent_ratio: average ratio a_c / a_p (given both active) (1D array)
    
    Measures whether children tend to have larger/smaller magnitudes than parent when co-active.
    """
    
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        parent_active_mask = cluster_data.parent_binary
        num_children = cluster_data.num_children
        
        fraction_parent_larger_than_child = np.full(num_children, np.nan)
        avg_ratio_arr = np.full(num_children, np.nan)
        
        for child_idx in range(num_children):
            child_acts = cluster_data.children_acts[:, child_idx]
            child_binary = cluster_data.children_binary[:, child_idx]
            
            # Restrict to datapoints where both p>THRESHOLD and c>THRESHOLD
            both_active_mask = parent_active_mask & child_binary
            
            if both_active_mask.sum() >= self.MIN_SAMPLES:
                parent_acts_both_active = cluster_data.parent_acts[both_active_mask]
                child_acts_both_active = child_acts[both_active_mask]
                
                # Percentage where parent > child
                parent_larger = parent_acts_both_active > child_acts_both_active
                fraction_parent_larger_than_child[child_idx] = parent_larger.mean()
                
                # Average ratio of child / parent
                # Avoid division by zero (though parent should be > threshold)
                ratios = child_acts_both_active / parent_acts_both_active
                avg_ratio_arr[child_idx] = ratios.mean()
        
        return {
            'fraction_parent_larger_than_child': fraction_parent_larger_than_child,
            'avg_child_parent_ratio': avg_ratio_arr
        }


class IndividualActiveMagnitudeMetric(IntraClusterActsMetric):
    """
    Computes mean activations when parent/children are individually active.
    
    Returns:
    - mean_child_when_active: mean activation of each child when that child is active (1D array)
    - mean_parent_when_active: mean activation of parent when parent is active (scalar)
    - child_parent_mean_ratio: ratio of child mean (when active) to parent mean (when active) (1D array)
    
    Measures typical magnitude of parent and children in their respective active states.
    """
    
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        parent_active_mask = cluster_data.parent_binary
        num_children = cluster_data.num_children
        
        # Mean parent activation when parent is active
        if parent_active_mask.sum() >= self.MIN_SAMPLES:
            mean_parent_when_active = cluster_data.parent_acts[parent_active_mask].mean()
        else:
            mean_parent_when_active = np.nan
        
        # Mean child activations when each child is individually active
        mean_child_when_active_arr = np.full(num_children, np.nan)
        child_parent_mean_ratio_arr = np.full(num_children, np.nan)
        
        for child_idx in range(num_children):
            child_acts = cluster_data.children_acts[:, child_idx]
            child_binary = cluster_data.children_binary[:, child_idx]
            
            if child_binary.sum() >= self.MIN_SAMPLES:
                mean_child_when_active_arr[child_idx] = child_acts[child_binary].mean()
                
                # Compute ratio if parent mean is valid
                if not np.isnan(mean_parent_when_active) and mean_parent_when_active > 0:
                    child_parent_mean_ratio_arr[child_idx] = mean_child_when_active_arr[child_idx] / mean_parent_when_active
        
        return {
            'mean_child_when_active': mean_child_when_active_arr,
            'mean_parent_when_active': mean_parent_when_active,
            'child_parent_mean_ratio': child_parent_mean_ratio_arr
        }

