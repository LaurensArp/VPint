

import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

from preprocessing.label import get_label_at_xy



def initialise_MRP(G,default_E=1):
    # Create initial MRP graph from G
    
    MRP = G.copy()
    for n in MRP.nodes(data=True):
        # Set E(i) to be equal to known observation if known, 0 otherwise (smooth 0 for zero division)
        # Smoothing not needed anymore as there is no more scaling, also set to 1 now
        E = 1
        if(not(n[1]['hidden'])):
            E = n[1]['label']
            
        nx.set_node_attributes(MRP,{n[0]:E},'E')
        
    return(MRP)

def initialise_W(G,default_weight=0.5):
    # Initialise MRP discount weight dict
    
    W = {}
    for n1,n2,w in G.edges(data=True):
        W[(n1,n2)] = default_weight
        
    return(W)
    
    
    
def run_MRP(G,W,num_iterations,debug=False):
    # Function for running MRP 
    
    for i in range(0,num_iterations):
        for node in G.nodes(data=True):
            # Compute state values: E(i) = b + 1/indeg(i) * (sum(E(j) for j in in(i))
            
            node_id = node[0]
            state_value = node[1]['E']
            
            new_state_value = 0
            summed_action_values = 0
            
            if(not(node[1]['hidden'])):
                # If value is known, keep E stable
                new_state_value = node[1]['label']
            else:
                # Iterate over actions
                for n1,n2,w in G.in_edges(node_id,data=True):
                    gamma = W[(n1,n2)]
                    destination_node = G.nodes(data=True)[n1]
                    expected_future_reward = destination_node['E']
                    action_value = gamma*expected_future_reward
                    summed_action_values += action_value
                    
                    if(debug):
                        print("Destination node: ",destination_node)
                        print("Expected reward: ",expected_future_reward)
                        print("Gamma: ",gamma)
                        print("Action value: ",action_value)

                # Compute state value as average of action values
                new_state_value = (summed_action_values / len(G.in_edges(node_id)))
                
                if(debug):
                    print("\nNew state value: ",new_state_value)
                    print("\n\n\n")

            # Set new state value
            nx.set_node_attributes(G, {node_id:new_state_value}, 'E')
        
    return(G)


   
def flip_hidden(G):
    for n in G.nodes(data=True):
        if(n[1]['hidden']):
            nx.set_node_attributes(G,{n[0]:False},'hidden')
        else:
            nx.set_node_attributes(G,{n[0]:True},'hidden')
        
            