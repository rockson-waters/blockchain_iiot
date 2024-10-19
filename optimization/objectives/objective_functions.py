from itertools import combinations
from typing import Dict, List, Tuple
import numpy as np
from optimization.node_generator import get_node_by_id, get_latency, get_throughput
from optimization.node_generator import Node as Peer
from optimization.objectives.constants import *




sim_data = {}

def initialize_variables(vars:dict):
    global sim_data
    sim_data = vars

def check_and_fix_solution(solution:np.ndarray):
    
    solution = solution.astype(int)
    solution = solution.clip(sim_data["lb"], sim_data["ub"]) 
    b = set(solution)

    if b.__contains__(sim_data["current_node_id"]):
        b.remove(sim_data["current_node_id"])

    while ((len(solution) - len(b)) > 0):
        b.add(np.random.randint(0, sim_data["number_of_nodes"], 1)[0])
        if b.__contains__(sim_data["current_node_id"]):
            b.remove(sim_data["current_node_id"])
    return np.array(list(b))



def generate_position(lb=None, ub=None):
    a = np.random.randint(0, sim_data["number_of_nodes"], sim_data["dim"])
    return a

def amend_position(solution:np.ndarray, lb=None, ub=None) -> list:

    sol = check_and_fix_solution(solution)
    return sol



def block_probability(solution:List[Tuple[Peer, Peer]]) -> float:
    
    block_prob = 0.0
    for j in solution:
        p1 = j[0] # The dictionary's value is a blockchain node
        p2 = j[1] # The dictionary's value is a blockchain node
        block_prob += p1.block_probability * p2.block_probability

    return block_prob
        

def average_delay(solution:List[Peer], current_node:Peer) -> float:
    
    n = len(solution)
    delay = 0.0
    
    for peer in solution:
        bw12 = get_throughput(current_node, peer)
        rtt12 = get_latency(current_node, peer)
        delay += BLOCK_SIZE/bw12 + rtt12*(BLOCK_SIZE/MSS)
    
    av_delay = (1 / n) * delay
    return av_delay


def processing_time_transactions(solution:List[Peer]) -> float:
    proc_time = 0.0

    for peer in solution:
        cn = int(peer.is_mining)
        cc = peer.compute_capacity
        proc_time += cn * (1 - np.exp(-cc / sim_data["c_max"])) * (1 / (ATS - ATA))
    return proc_time


# Define Fitness Function

def fitness_multi(solution:np.ndarray):
    neighbours:List[Peer] = []
    if (solution.__contains__(sim_data["current_node_id"])):
        solution = check_and_fix_solution(solution)
    for node_id in solution:
        node = get_node_by_id(node_id)
        neighbours.append(node)
    current_node = get_node_by_id(sim_data["current_node_id"])
    list_combinations = list(combinations(neighbours, 2))

    def obj1(neigh_pairs:List[Tuple[Peer, Peer]]):
        return block_probability(neigh_pairs)
    
    def obj2(neigh:np.array, cur_node:Peer):
        return -average_delay(neigh, cur_node) # in oder to convert this objective to a maximization, I negated it

    def obj3(neigh:np.array):
        return processing_time_transactions(neigh)

    return [obj1(list_combinations), obj2(neighbours, current_node), obj3(neighbours)]

