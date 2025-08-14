import networkx as nx
import numpy as np
import os

def get_dataset_path(filename):
    """Get the correct path to dataset file"""
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, filename)

def load_karate():
    G = nx.karate_club_graph()
    ground_truth = np.array([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes()])
    return G, ground_truth

def load_dolphins():
    G = nx.read_gml(get_dataset_path('dolphins.gml'), label='id')
    ground_truth = None
    return G, ground_truth

def load_football():
    G = nx.read_gml(get_dataset_path("football.gml"))
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    ground_truth = [G.nodes[n].get("value", 0) for n in G.nodes()]
    return G, ground_truth

def load_email():
    """Load Email-Eu-core network"""
    G = nx.Graph()
    
    # Read edges from file
    with open(get_dataset_path('email-Eu-core.txt'), 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        G.add_edge(u, v)
                    except ValueError:
                        continue
    
    # No ground truth available
    ground_truth = None
    return G, ground_truth

def load_facebook():
    """Load Facebook combined network"""
    G = nx.Graph()
    
    # Read edges from file
    with open(get_dataset_path('facebook_combined.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
                except ValueError:
                    continue
    
    # No ground truth available
    ground_truth = None
    return G, ground_truth