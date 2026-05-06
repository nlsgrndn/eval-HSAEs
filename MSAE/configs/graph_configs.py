import os
from configs.conditional_act_graph_config import get_conditional_act_graph_config
from configs.mcs_graph_config import get_mcs_graph_config
from configs.weighted_conditional_act_graph_config import get_weighted_conditional_act_graph_config

def get_graph_config():
    graph_type = os.environ.get('GRAPH_TYPE', "wondact")
    if "mcs" in graph_type:
         return get_mcs_graph_config()
    elif "condact" in graph_type:
        return get_conditional_act_graph_config()
    elif "wondact" in graph_type:
        return get_weighted_conditional_act_graph_config()
    else:
        raise ValueError(f"Unsupported GRAPH_TYPE: {graph_type}.")

def get_graph_name():
    return get_graph_config().name