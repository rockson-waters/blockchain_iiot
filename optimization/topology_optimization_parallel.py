from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from mealpy.swarm_based import PSO
from optimization.mst import Graph
from optimization.node_generator import *
from optimization.objectives.objective_functions import amend_position, fitness_multi, generate_position
from optimization.objectives.constants import *
from optimization.objectives.utilities import initialize_sim_data, reinitialize_sim_data, save_all_solutions_to_file, reset_model_history, plot_peer_solution, save_runtime_to_file
from multiprocessing import Process
# from numba import jit


MST = True

current_node_id = 0
global number_of_nodes 
number_of_nodes = 400
num_consensus_nodes = ceil(0.15 * number_of_nodes)
num_of_regions = 5
solutions = {}
total_runtime = []
total_runtime_dict = {}
total_runtime_per_node_dict = {}

sim_data = {}
all_nodes
nf = NodeFactory(num_of_regions, number_of_nodes, num_consensus_nodes)

sim_data = initialize_sim_data(sim_data, number_of_nodes, all_nodes)

g = Graph(number_of_nodes)
graph = g.read_latencies()
g.graph = graph
g.prims()
g.get_adjacency_matrix()
g.save_solution(number_of_nodes)


# Define Problem
problem_multi = {
    "fit_func": fitness_multi,
    "lb": sim_data["lb"],
    "ub": sim_data["lb"],
    "minmax": "max",
    "obj_weights": [1, 1, 1],
    "generate_position": generate_position,
    "amend_position": amend_position,
    "log_to": None,
}
# Define termination criteria
term_dict = {
"mode": "ES",
"quantity": 30  # after 30 epochs, if the global best doesn't improve then we stop the program
}


for n in [number_of_nodes]:
    number_of_nodes = n
    problem_multi["lb"] = sim_data["lb"] # update dimensions in problem definnition
    problem_multi["ub"] = sim_data["ub"] # update dimensions in problem definnition
    graph_generation_time = 0
    
    reinitialize_sim_data(sim_data, number_of_nodes, current_node_id)
    pso_model = PSO.BasePSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)

    for m in range(0, number_of_nodes, 1):
        current_node_id = m
        sim_data["current_node_id"] = current_node_id
        best_position, best_fitness_value = pso_model.solve()
        
        solutions.update({current_node_id: best_position.tolist()})
        model = PSO.BasePSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        # model = PSO.C_PSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        # model = PSO.CL_PSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        # model = PSO.HPSO_TVAC(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        # model = PSO.PPSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        b = np.array(pso_model.history.list_epoch_time)
        b = np.sum(b)
        total_runtime_per_node_dict[f"PSO_default_total_{number_of_nodes}_current_{current_node_id}"] = b
        graph_generation_time += b
        reset_model_history(pso_model)
        
    # save runtime to log
    save_runtime_to_file(total_runtime_per_node_dict, "one", number_of_nodes)
    total_runtime_per_node_dict = {}
    total_runtime_dict[number_of_nodes] = graph_generation_time
    total_runtime.append(graph_generation_time)
    
save_runtime_to_file(total_runtime_dict, "all", number_of_nodes)
save_all_solutions_to_file(solutions, number_of_nodes)
save_location_names(nf.locs)

