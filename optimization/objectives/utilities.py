import json
from typing import Dict, List, Union
from mealpy.swarm_based.PSO import BasePSO
import numpy as np

from optimization.node_generator import Node
from optimization.objectives.objective_functions import initialize_variables

def get_max_compute_capacity(nodes:Dict[int, Node]):
    ccs = []
    for node in nodes.values():
        ccs.append(node.compute_capacity)
    max_val = max(ccs)
    return max_val

def get_average_number_of_neighbours(num_nodes:int) -> int:
        n = num_nodes
        M = ((n - 1) / n) * np.log2(n)
        return int(np.ceil(M))

def save_runtime_to_file(runtime:Union[Dict, List], name:str="runtime", num_of_nodes=100, run_id=1,algo=""):
    
    with open(f"out/{num_of_nodes}/{run_id}_{algo}_{name}.json", 'w') as f:
        f.write(json.dumps(runtime))
    

def save_all_solutions_to_file(solution:Dict, num_nodes=100, run_id=1, algo=""):
    with open(f"out/{num_nodes}/{run_id}_{algo}_solution.json", 'w') as f:
        f.write(json.dumps(solution))

def plot_peer_solution(model:BasePSO, total_nodes:int, node_id:int, algorithm:str="PSO"):
    
    model.history.save_diversity_chart(filename=f"out/total_nodes_{total_nodes}/node_id_{node_id}/diversity_chart_{algorithm}")
    model.history.save_exploration_exploitation_chart(filename=f"out/total_nodes_{total_nodes}/node_id_{node_id}/exploration_exploitation_{algorithm}")
    model.history.save_global_best_fitness_chart(filename=f"out/total_nodes_{total_nodes}/node_id_{node_id}/global_best_fitness_{algorithm}")
    model.history.save_global_objectives_chart(filename=f"out/total_nodes_{total_nodes}/node_id_{node_id}/global_objectives_{algorithm}")
    model.history.save_local_best_fitness_chart(filename=f"out/total_nodes_{total_nodes}/node_id_{node_id}/local_best_fitness_{algorithm}")
    model.history.save_local_objectives_chart(filename=f"out/total_nodes_{total_nodes}/node_id_{node_id}/local_objectives_{algorithm}")
    model.history.save_runtime_chart(filename=f"out/total_nodes_{total_nodes}/node_id_{node_id}/runtime_{algorithm}")
    model.history.save_trajectory_chart(filename=f"out/total_nodes_{total_nodes}/node_id_{node_id}/trajectory_{algorithm}")

def reset_model_history(model:BasePSO):
    model.history.list_epoch_time = []
    model.history.list_current_best = []
    model.history.list_current_best_fit = []
    model.history.list_diversity = []
    model.history.list_exploitation = []
    model.history.list_global_best = []
    model.history.list_global_best_fit = []
    model.history.list_population = []
    model.history.list_exploration = []
    return model

def initialize_sim_data(sim_data:Dict, num_nodes:int, all_peers:dict):

    sim_data["number_of_nodes"] = num_nodes
    sim_data["dim"] = get_average_number_of_neighbours(sim_data["number_of_nodes"])
    sim_data["lb"] = [0] * sim_data["dim"] # Lower bound
    sim_data["ub"] = [sim_data["number_of_nodes"] - 1] * sim_data["dim"] # Upper bound
    sim_data["c_max"] = get_max_compute_capacity(all_peers) # maximum compute capacity in the network
    sim_data["current_node_id"] = 0
    initialize_variables(sim_data)
    return sim_data
    
def reinitialize_sim_data(sim_data:Dict, num_nodes:int, current_node_id:int):

    sim_data["number_of_nodes"] = num_nodes
    sim_data["dim"] = get_average_number_of_neighbours(sim_data["number_of_nodes"])
    sim_data["lb"] = [0] * sim_data["dim"] # Lower bound
    sim_data["ub"] = [sim_data["number_of_nodes"] - 1] * sim_data["dim"] # Upper bound
    #sim_data["c_max"] = get_max_compute_capacity(all_peers) # maximum compute capacity in the network
    sim_data["current_node_id"] = current_node_id
    initialize_variables(sim_data)
    return sim_data
    
