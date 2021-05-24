import pandas as pd
import numpy as np
import networkx as nx














def graph_to_tensor_train(G,width,height):
    
    # Every pixel has grid:
    #     instance_width / 2 left, instance_width / 2 right, instance_height / 2 up, instance_height / 2 down
    #     attributes of length len(A) + 1 (for labels if available)
    #     use mean imputation where not available
   
    attr_list = list(nx.get_node_attributes(G,'A').values())
    num_features = len(attr_list[0])
    
    x_list = list(nx.get_node_attributes(G,'x').values())
    max_x = max(x_list)
    
    y_list = list(nx.get_node_attributes(G,'y').values())
    max_y = max(y_list)
    
    hidden_list = list(nx.get_node_attributes(G,'hidden').values())
    num_instances = len(G.nodes) - sum(hidden_list) # Only train on non-hidden nodes
    
    
    # Mean label
    
    label_list = list(nx.get_node_attributes(G,'label').values())
    mean_label = np.mean(y_list)
    
    # Mean A vector
    
    mean_A = np.zeros(num_features)
    attr_matrix = np.array(attr_list) # rows should be instances, cols attributes
    
    for i in range(0,len(mean_A)):
        mean_A[i] = np.mean(attr_matrix[:,i])
        
        
    # Create 3d grid for easy indexing (can't search for nodes by x/y)
    
    grid = np.zeros((max_y+1,max_x+1,num_features+1))
    
    for n in G.nodes(data=True):
        r = n[1]['y']
        c = n[1]['x']
        if(n[1]['hidden']):
            y_val = mean_label
        else:
            y_val = n[1]['label']
        vec = np.append(n[1]['A'],[y_val])
        
        grid[r,c,:] = vec
        
    # Set up training set
    
    X = np.zeros((num_instances,height*2+1,width*2+1,num_features+1)) # (i,y,x,a), height*2 (up and down) +1 (center)
    y_train = np.zeros(num_instances)

    # Get X,y

    i = 0
    for n in G.nodes(data=True):
        if(not(n[1]['hidden'])):
            # Target pixel indices in grid
            r = n[1]['y']
            c = n[1]['x']
            
            y_train[i] = n[1]['label']
            
            # y/c for offset on r/c
            for y in range(1,height+1): # 1 to h+1 because we don't want 0 indexing
                for x in range(1,width+1):
                    ind1 = (r-y,c-x) # top-left
                    ind2 = (r+y,c-x) # bottom-left
                    ind3 = (r-y,c+x) # top-right
                    ind4 = (r+y,c+x) # bottom-right
                    ind5 = (r,c-x) # center-left
                    # ignore center-center
                    ind6 = (r,c+x) # center-right
                    ind6 = (r+y,c) # bottom-center
                    ind7 = (r-y,c) # top-center
                    ind = [ind1,ind2,ind3,ind4,ind5,ind6,ind7]
                    
                    for ind_n in ind:
                        #print("(r,c) = (" + str(r) + "," + str(c) + ")")
                        #print("(y,x) = (" + str(y) + "," + str(x) + ")")
                        #print("ind_n = (" + str(ind_n[0]) + "," + str(ind_n[1]) + ")")
                        
                        
                        
                        # If expected surrounding pixel out of bounds, replace all by mean
                        if(ind_n[0] < 0 or ind_n[0] >= grid.shape[0] or 
                          ind_n[1] < 0 or ind_n[1] >= grid.shape[1]):
                            vec = np.append(mean_A,[mean_label])
                            
                        # If in bounds, use grid vector
                        else:
                            vec = grid[ind_n[0],ind_n[1],:]
                            
                        # X uses x/y relative to target pixel, not absolute grid index
                        # So recompute raw x/y instead of using ind
                        X[i,ind_n[0]-r,ind_n[1]-c,:] = vec
        
            i += 1
            

    return(X,y_train)



def graph_to_tensor_test(G,width,height):
    
    # Every pixel has grid:
    #     instance_width / 2 left, instance_width / 2 right, instance_height / 2 up, instance_height / 2 down
    #     attributes of length len(A) + 1 (for labels if available)
    #     use mean imputation where not available
   
    attr_list = list(nx.get_node_attributes(G,'A').values())
    num_features = len(attr_list[0])
    
    x_list = list(nx.get_node_attributes(G,'x').values())
    max_x = max(x_list)
    
    y_list = list(nx.get_node_attributes(G,'y').values())
    max_y = max(y_list)
    
    hidden_list = list(nx.get_node_attributes(G,'hidden').values())
    num_instances = sum(hidden_list) # Only test on hidden nodes
    
    
    # Mean label
    
    label_list = list(nx.get_node_attributes(G,'label').values())
    mean_label = np.mean(y_list)
    
    # Mean A vector
    
    mean_A = np.zeros(num_features)
    attr_matrix = np.array(attr_list) # rows should be instances, cols attributes
    
    for i in range(0,len(mean_A)):
        mean_A[i] = np.mean(attr_matrix[:,i])
        
        
    # Create 3d grid for easy indexing (can't search for nodes by x/y)
    
    grid = np.zeros((max_y+1,max_x+1,num_features+1))
    
    for n in G.nodes(data=True):
        r = n[1]['y']
        c = n[1]['x']
        if(n[1]['hidden']):
            y_val = mean_label
        else:
            y_val = n[1]['label']
        vec = np.append(n[1]['A'],[y_val])
        
        grid[r,c,:] = vec
        
    # Set up training set
    
    X = np.zeros((num_instances,height*2+1,width*2+1,num_features+1)) # (i,y,x,a), height*2 (up and down) +1 (center)
    y_train = np.zeros(num_instances)

    # Get X,y

    i = 0
    for n in G.nodes(data=True):
        if(n[1]['hidden']):
            # Target pixel indices in grid
            r = n[1]['y']
            c = n[1]['x']
            
            y_train[i] = n[1]['label']
            
            # y/c for offset on r/c
            for y in range(1,height+1): # 1 to h+1 because we don't want 0 indexing
                for x in range(1,width+1):
                    ind1 = (r-y,c-x) # top-left
                    ind2 = (r+y,c-x) # bottom-left
                    ind3 = (r-y,c+x) # top-right
                    ind4 = (r+y,c+x) # bottom-right
                    ind5 = (r,c-x) # center-left
                    # ignore center-center
                    ind6 = (r,c+x) # center-right
                    ind6 = (r+y,c) # bottom-center
                    ind7 = (r-y,c) # top-center
                    ind = [ind1,ind2,ind3,ind4,ind5,ind6,ind7]
                    
                    for ind_n in ind:
                        #print("(r,c) = (" + str(r) + "," + str(c) + ")")
                        #print("(y,x) = (" + str(y) + "," + str(x) + ")")
                        #print("ind_n = (" + str(ind_n[0]) + "," + str(ind_n[1]) + ")")
                        
                        
                        
                        # If expected surrounding pixel out of bounds, replace all by mean
                        if(ind_n[0] < 0 or ind_n[0] >= grid.shape[0] or 
                          ind_n[1] < 0 or ind_n[1] >= grid.shape[1]):
                            vec = np.append(mean_A,[mean_label])
                            
                        # If in bounds, use grid vector
                        else:
                            vec = grid[ind_n[0],ind_n[1],:]
                            
                        # X uses x/y relative to target pixel, not absolute grid index
                        # So recompute raw x/y instead of using ind
                        X[i,ind_n[0]-r,ind_n[1]-c,:] = vec
        
            i += 1
            

    return(X,y_train)





