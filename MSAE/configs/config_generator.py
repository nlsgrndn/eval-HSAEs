
from configs.graph_configs import get_graph_name, get_graph_config
from configs.mcs_graph_config import get_mcs_graph_config
from configs.weighted_conditional_act_graph_config import get_weighted_conditional_act_graph_config, WeightedConditionalActivationGraphConfig
from configs.conditional_act_graph_config import get_conditional_act_graph_config, ConditionalActivationGraphConfig
from configs.activation_preprocessing import get_acts_preprocess_cfg
import copy

def get_same_configs_with_mcs_graph(configs):
    modified_configs = []
    for config in configs:
        config_copy = copy.deepcopy(config)
        config_copy.graph_name = get_mcs_graph_config().name
        modified_configs.append(config_copy)
    return modified_configs

def get_same_configs_with_wondact_graph(configs):
    modified_configs = []
    for config in configs:
        config_copy = copy.deepcopy(config)
        config_copy.graph_name = get_weighted_conditional_act_graph_config().name
        modified_configs.append(config_copy)
    return modified_configs


def get_same_configs_with_default_graph_using_th1_activation_preprocessing(configs):
    modified_configs = []
    for config in configs:
        config_copy = copy.deepcopy(config)
        graph_config = get_graph_config()
        acts_preprocess_cfg = get_acts_preprocess_cfg("th1_maxNS500000_MC") 
        graph_config.metrics_config = acts_preprocess_cfg
        config_copy.graph_name = graph_config.name
        modified_configs.append(config_copy)
    return modified_configs

def get_same_configs_with_modified_graph_creation_thresholds(configs, new_thresholds:list):
    modified_configs = []
    for config in configs:
        for new_threshold in new_thresholds:
            config_copy = copy.deepcopy(config)
            graph_config = get_graph_config()
            assert isinstance(graph_config, ConditionalActivationGraphConfig) or isinstance(graph_config, WeightedConditionalActivationGraphConfig), "Expected graph config to be ConditionalActivationGraphConfig"
            graph_config.conditional_activation_parent_given_child_threshold = new_threshold
            graph_config.conditional_activation_child_given_parent_threshold = new_threshold
            config_copy.graph_name = graph_config.name
            modified_configs.append(config_copy)
    return modified_configs

def get_same_configs_with_default_graph_using_nonMC_activation_preprocessing(configs):
    modified_configs = []
    for config in configs:
        config_copy = copy.deepcopy(config)
        config_copy.graph_name = config_copy.graph_name.replace("_MC_", "_")
        modified_configs.append(config_copy)
    return modified_configs