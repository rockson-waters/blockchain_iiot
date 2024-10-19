from math import ceil
from multiprocessing import Process
import multiprocessing
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mealpy.swarm_based import PSO      
from optimization.mst import Graph
from optimization.node_generator import *
from optimization.objectives.objective_functions import amend_position, fitness_multi, generate_position
from optimization.objectives.constants import *
from optimization.objectives.utilities import initialize_sim_data, reinitialize_sim_data, save_all_solutions_to_file, reset_model_history, plot_peer_solution, save_runtime_to_file
# from numba import jit


# current_node_id = 0
network_sizes = [10, 20, 30, 40, 50]
RUNS_PER_SIZE = 10
global number_of_nodes 
# number_of_nodes = network_sizes[0]
# num_consensus_nodes = ceil(0.15 * number_of_nodes)
num_of_regions = 5
# solutions = {} # used to store solutions obtained by a parallel worker
# solutions_all = {} # used to store solutions from all parallel workers
# total_runtime = []
# total_runtime_dict = {}
# total_runtime_per_node_dict = {}
# sim_data = {}
# all_nodes
PARALLEL_WORKERS_COUNT = 12
CONSENSUS_NODES_PERCENT = 0.15




def find_mst(num_nodes, run_id=0):
    g = Graph(num_nodes)
    graph = g.read_latencies()
    g.graph = graph
    start_time = time()
    g.prims()
    duration = time() - start_time
    # print(duration)
    g.get_adjacency_matrix()
    g.save_solution(num_nodes, run_id)

def calculate_optimum_solution(num_workers, problem_dict:dict, term_dict:dict ,graph_generation_time, sim_data, all_soln, algo_model):
    start = 0
    end = 0
    process_list:List[Process] = []

    interval = int(number_of_nodes / num_workers)
    rem = int(number_of_nodes % num_workers)
    num_per_worker = [interval] * num_workers

    y = 0
    while (rem > 0):
        num_per_worker[y] += 1
        y += 1
        rem -= 1

    start = 0
    for i in range(num_workers):
        num = num_per_worker[i]
        end = start + num
        ranges = range(start, end)
        # model = PSO.BasePSO(problem=problem_dict, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        # print(type(model))
        # model = PSO.C_PSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        # model = PSO.CL_PSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        # model = PSO.HPSO_TVAC(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        # model = PSO.PPSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
        process = Process(target=_find_ONS, args=(i, ranges, algo_model, graph_generation_time, sim_data, all_soln ))
        process_list.append(process)
        start = end
    
    start_time = time()
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()
    
    duration = time() - start_time
    # print(f"Duration: {duration}")



def _find_ONS(worker_id:str, nodes_list:range, model:PSO.BasePSO, graph_generation_time:multiprocessing.Array, sim_data:dict, all_soln:dict):
    graph_generation_time[worker_id] = 0
    solutions = {}
    for m in nodes_list:
        sim_data["current_node_id"] = m
        best_position, best_fitness_value = model.solve()       
        solutions.update({m: best_position.tolist()})
        b = sum(model.history.list_epoch_time)
        graph_generation_time[worker_id] += b
        reset_model_history(model)
    all_soln.update(solutions)
    # print(solutions.items())



def find_runtimes(runtimes:Dict[str,float]):
    return max(runtimes)




with multiprocessing.Manager() as manager:
    for n in network_sizes:
        for m in range(RUNS_PER_SIZE):
            number_of_nodes = n
            num_consensus_nodes = ceil(CONSENSUS_NODES_PERCENT * number_of_nodes)
            current_node_id = 0
            solutions = {} # used to store solutions obtained by a parallel worker
            total_runtime = []
            total_runtime_dict = {}
            total_runtime_per_node_dict = {}
            sim_data = {}
            all_nodes
            nf = NodeFactory(num_of_regions, number_of_nodes, num_consensus_nodes, m)
            sim_data = initialize_sim_data(sim_data, number_of_nodes, all_nodes)
            graph_generation_time = multiprocessing.Array("d", PARALLEL_WORKERS_COUNT)

            find_mst(n, m)

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

            
            models = {}
            models["BasePSO"] = PSO.BasePSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
            # models["C_PSO"] = PSO.C_PSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
            models["CL_PSO"] = PSO.CL_PSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
            models["HPSO"] = PSO.HPSO_TVAC(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
            models["PPSO"] = PSO.PPSO(problem=problem_multi, epoch=EPOCH, pop_size=POP_SIZE, termination=term_dict)
            # print(models[0])

            for algo, model in models.items():
                solutions_all = manager.dict({})
                calculate_optimum_solution(PARALLEL_WORKERS_COUNT, problem_multi, term_dict, graph_generation_time, sim_data, solutions_all, model)
                # print(f"Runtime: {find_runtimes(graph_generation_time)}")
                solutions_all
                sol = sorted(solutions_all.items(), key = lambda x:x[0])
                
                save_runtime_to_file({number_of_nodes: find_runtimes(graph_generation_time)}, num_of_nodes=n, run_id=m, algo=algo)
                # save_runtime_to_file(total_runtime_dict, "all", number_of_nodes)
                save_all_solutions_to_file(dict(sol), number_of_nodes, run_id=m, algo=algo)
                # save_location_names(nf.locs)