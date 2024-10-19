from itertools import combinations
from math import ceil
from typing import Dict, List
import numpy as np
import scipy.stats
import json


inter_region_latency_dist = "norm" # average between 50 and 200, sd between 0.08 and 0.3
inter_region_throughput_dist = "beta" # 0.4-0.6; 0.3-0.5; 30-60; 30-80;

intra_region_latency_dist = "invgamma" # 4-15; 0.3-0.9; 0.9-3;
intra_region_throughput = "invgamma" # 100-300; -400 to -100; 70000-170000

latencies = {}
throughputs = {}
all_nodes = {} # stores node objects with their unique ids as the key
node_id_name = {} # stores node's integer id against its name
node_properties = {} # stores other node properties


rng = np.random.default_rng()


class Node:
    def __init__(self, node_id:str, is_mining=False, compute_capacity:float=0.0, location:str=""):
            self.node_id = node_id
            self.node_id_num = 0 # unique numerical node identifier in the network
            self.is_mining = is_mining
            self.block_probability = 0.0
            self.compute_capacity = compute_capacity
            self.location = location

def get_latency(node_1:Node, node_2:Node):
    latency = latencies.get(f"{node_1.node_id}_{node_2.node_id}")
    if latency is None:
        latency = latencies.get(f"{node_2.node_id}_{node_1.node_id}")
    return float(latency)

def get_throughput(node_1:Node, node_2:Node):
    throughput = throughputs.get(f"{node_1.node_id}_{node_2.node_id}")
    if throughput is None:
        throughput = throughputs.get(f"{node_2.node_id}_{node_1.node_id}")
    return float(throughput)

def get_random_values(distribution: dict, n=1):
        """Receives a `distribution` and outputs `n` random values
        Distribution format: { \'name\': str, \'parameters\': tuple }"""
        dist = getattr(scipy.stats, distribution['name'])
        param = distribution['parameters'] # make_tuple(distribution['parameters'])
        num = dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
        return num[0]

def generate_latencies(distribution:dict, nodes:List[Node]):
    
    all_com = combinations(nodes, 2)
    list_comb = list(all_com)

    for pair in list_comb:
        a = pair[0]
        b = pair[1]
        latency = get_random_values(distribution)
        latencies[f"{a.node_id}_{b.node_id}"] = latency
    return latencies

def generate_throughputs(distribution:dict, nodes:List[Node]):
    
    all_com = combinations(nodes, 2)
    list_comb = list(all_com)
    
    for pair in list_comb:
        a = pair[0]
        b = pair[1]
        throughput = get_random_values(distribution)
        throughputs[f"{a.node_id}_{b.node_id}"] = throughput
    return throughputs

def save_latency(run_id:int, lat:dict, path:str, num = 100):
    with open(f"{path}/{num}/{run_id}_latencies.json", 'w') as f:
        f.write(json.dumps(lat))

def save_throughputs(run_id:int, th:dict, path:str, num = 100):
    with open(f"{path}/{num}/{run_id}_throughputs.json", 'w') as f:
        f.write(json.dumps(th))

def save_node_properties(run_id:int, th:dict, path:str, num = 100):
    with open(f"{path}/{num}/{run_id}_node_properties.json", 'w') as f:
        f.write(json.dumps(th))

def _read_json_file(file_location:str):
        with open(file_location) as f:
            return json.load(f)


class Location:

    def __init__(self, name:str, size:int):
        """Creates a location and initialises it with probability distribution parameters

        Args:
            name (str): name of the location
            size (int): the number of nodes to generate
        """
        self.name = name
        self.size = size
        self.nodes:List[Node] = []
        self.distribution_latency = {}
        self.distribution_throughput = {}

        self.distribution_latency["name"] = intra_region_latency_dist
        self.distribution_latency["parameters"] = (rng.uniform(low=4, high=15), rng.uniform(low=0.3, high=0.9), rng.uniform(low=0.9, high=3))
        self.distribution_throughput["name"] = intra_region_throughput
        self.distribution_throughput["parameters"] = (rng.uniform(low=100, high=300), rng.uniform(low=100, high=400), rng.uniform(low=70000, high=170000))
        
        self.generate_nodes()
        self.generate_node_latencies()
        self.generate_node_throughputs()
        self.generate_compute_capacities()
        self.generate_block_probabilities()

    def generate_nodes(self):
        
        for i in range(0, self.size):
            node = Node(f"{self.name}_{i}", location=self.name)
            self.nodes.append(node)


    def generate_node_latencies(self):
        l = generate_latencies(self.distribution_latency, self.nodes)
        return l

    def generate_node_throughputs(self):
        t = generate_throughputs(self.distribution_throughput, self.nodes)
        return t
    
    def generate_compute_capacities(self):
        """Generates the computes capacity of each and every peer"""
        for node in self.nodes:
            cc = rng.uniform(low=20, high=40)
            node.compute_capacity = float(cc)
        return self.nodes

    def generate_block_probabilities(self):
        """Generates the probability of receiving a block from eny peer"""
        probs = rng.uniform(low=1, high=100, size=len(self.nodes))
        probs = np.array(probs) #* self.consensus_nodes
        probs = probs / probs.sum()
        
        i = 0
        for node in self.nodes:
            node.block_probability = probs[i]
            i += 1
        return self.nodes



class InterLocations:

    def __init__(self, locations:List[Location], run_id:int, num_mining_nodes:int=5):
        """Initialises a location with latency and bandwidth

        Args:
            name (str): name of the location
            dist (Distributions): the type of distribution to use
        """
        self.distribution_latency = {}
        self.distribution_throughput = {}
        self.locations = locations
        

        self.distribution_latency["name"] = inter_region_latency_dist
        
        self.distribution_throughput["name"] = inter_region_throughput_dist
        self.generate_inter_location_data(locations)
        assign_unique_id(locations) # Fills the all_nodes dictionary with its entries
        assign_mining_roles(all_nodes, num_mining_nodes)
        self.save_gen_latencies(run_id)
        self.save_gen_throughputs(run_id)
        self.save_gen_node_properties(run_id)

    def generate_inter_location_data(self, locations:List[Location]):
        all_combinations = combinations(locations, 2)
        list_combinations = list(all_combinations)
        
        for locs in list_combinations:
            loc1 = locs[0]
            loc2 = locs[1]
            
            rng = np.random.default_rng()
            self.distribution_latency["parameters"] = (rng.uniform(low=50, high=200), rng.uniform(low=0.08, high=0.3))
            self.distribution_throughput["parameters"] = (rng.uniform(low=0.4, high=0.6), rng.uniform(low=0.3, high=0.5), rng.uniform(low=30, high=60), rng.uniform(low=30, high=80))
            
            for node_loc1 in loc1.nodes:
                for node_loc2 in loc2.nodes:
                    latency = get_random_values(self.distribution_latency)
                    throughput = get_random_values(self.distribution_throughput)
                    latencies[f"{node_loc1.node_id}_{node_loc2.node_id}"] = latency
                    throughputs[f"{node_loc1.node_id}_{node_loc2.node_id}"] = throughput
        return latencies, throughputs
    
    def save_gen_latencies(self, run_id:int, folder_path:str="out"):
        save_latency(run_id, latencies, folder_path, len(all_nodes))

    def save_gen_throughputs(self, run_id:int, folder_path:str="out"):
        save_throughputs(run_id, throughputs, folder_path, len(all_nodes))

    def save_gen_node_properties(self, run_id:int, folder_path:str="out"):
        save_node_properties(run_id, node_properties, folder_path, len(all_nodes))
    
def save_location_names(locs:List[Location], path:str="out"):
    loc_names = []
    for loc in locs:
        loc_names.append(loc.name)
    
    with open(f"{path}/loc_names.json", 'w') as f:
        f.write(json.dumps(loc_names))
    

def get_all_nodes():
    return all_nodes

def assign_unique_id(locations:List[Location]):
    nodes:List[Node] = []
    for loc in locations:
        nodes += loc.nodes

    j = 0
    for node in nodes:
        node.node_id_num = j
        all_nodes.update({node.node_id_num: node})
        node_id_name.update({node.node_id_num: node.node_id})
        node_properties.update({node.node_id_num: {"node_id": node.node_id, "is_mining": node.is_mining, "block_probability":node.block_probability, "compute_capacity": node.compute_capacity, "location":node.location}})
        j += 1
    return nodes

def assign_mining_roles(nodes_dict:Dict[int, Node], n=5):
    rng = np.random.default_rng()
    selected = rng.integers(0, len(nodes_dict), n)

    for node_id in selected:
        node = nodes_dict[node_id]
        node.is_mining = True
        node_properties[node.node_id_num]["is_mining"] = node.is_mining
        # node_properties.update({node.node_id_num: {"compute_capacity": node.compute_capacity, "is_mining": node.is_mining}})
    return selected

def get_node_by_id(node_id:int):
    node:Node = all_nodes.get(node_id)
    return node

def get_node_by_name(node_name:str):
    node_id:int = 0
    for k,v in node_id_name.items():
        if node_name == v:
            node_id = k
            break
    node = get_node_by_id(node_id)
    return node


class NodeFactory:
    def __init__(self, num_regions:int=1, num_nodes:int=100, num_mining_nodes:int=5, run_id=0):
        peers_per_region = self.get_number_of_nodes_per_region(num_regions, num_nodes)
        self.locs = []
        self.num_nodes = num_nodes

        for i in range(0, num_regions):
            loc = Location(f"loc{i}", peers_per_region[i])
            self.locs.append(loc)
        
        inter_locs = InterLocations(self.locs, run_id, num_mining_nodes)
        inter_locs.save_gen_throughputs(run_id)
        inter_locs.save_gen_latencies(run_id)

    def number_of_messages(self):
        N = self.num_nodes
        n = int(ceil(((N - 1) / N) * np.log2(N)))
        av_msg = (28.8 * N * (2 + (28 * n))) + rng.integers(-100, 1000)
        return int(av_msg)

    def get_number_of_nodes_per_region(self, num_regions:int, num_nodes:int):
        
        lb = 3
        ub = int((num_nodes / 4) + 10)
        peer_nums = rng.integers(lb, ub, num_regions)
        
        while sum (peer_nums) > num_nodes:
            x = rng.integers(0, num_regions)
            if peer_nums[x] < lb:
                peer_nums[x] = rng.integers(lb, ub, 1)
            else:
                peer_nums[x] -= 1

        while sum(peer_nums) < num_nodes:
            peer_nums[rng.integers(0, num_regions)] += 1

        return peer_nums.tolist()

        
def read_node_values(folder_path:str="blocksim/out"):
    global latencies
    global throughputs
    global all_nodes
    global node_id_name
    global node_properties

    
    node_properties = dict(_read_json_file(f"{folder_path}/node_properties.json"))
    loc_names = list(_read_json_file(f"{folder_path}/loc_names.json"))
    latencies = dict(_read_json_file(f"{folder_path}/latencies.json"))
    throughputs = dict(_read_json_file(f"{folder_path}/throughputs.json"))
    # solutions = dict(_read_json_file(f"{folder_path}/solution.json"))

    all_nodes = {}
    node_id_name = {}
    for x,y in node_properties.items():
        node = Node(y.node_id, y.is_mining, y.compute_capacity, y.location)
        node_id_name[node.node_id_num] = node.node_id
        all_nodes[x] = node

    sim_data = {}
    sim_data["node_properties"] = node_properties
    sim_data["loc_names"] = loc_names
    sim_data["latencies"] = latencies
    sim_data["throughputs"] = throughputs
    # sim_data["solutions"] = solutions
    sim_data["number_of_nodes"] = len(sim_data["node_properties"])
    # sim_data["dim"] = get_average_number_of_neighbours(sim_data["number_of_nodes"])
    # sim_data["lb"] = [0] * sim_data["dim"] # Lower bound
    # sim_data["ub"] = [sim_data["number_of_nodes"] - 1] * sim_data["dim"] # Upper bound
    # sim_data["current_node_id"] = 0
    return sim_data
