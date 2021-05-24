
import geopandas as gpd
import pandas as pd


def apply_taxonomy(S,taxonomy):
    # Map specific types to higher-level categories, defaulting to "other"
    
    for i,r in S.iterrows():
        t = r['type']
        if(t in taxonomy):
            new_t = taxonomy[t]
        else:
            new_t = "other"
        S.at[i,'type'] = new_t

    return(S)
    

def find_types(S,optimisation_path,working_path,type_filter_method,type_params,taxonomy_filename=None,verbose=False):
    # Process types in the dataset, using selection/filter methods as
    # specified by the user
    
    type_frequency_ratio = type_params[0]
    type_top_n = type_params[1]
    type_top_n_percent = type_params[2]
    type_top_n_variable = type_params[3]
    
    types_temp = []
    types_dict = {}
    types = []

    taxonomy = {}

    # Create dict with counts of each type
    for t in S['type']:
        if(t in types_temp):
            types_dict[t] += 1
        else:
            types_temp.append(t)
            types_dict[t] = 1


    # Filter types

    if(type_filter_method == "frequency"):
        for k,v in types_dict.items():
            if(v/len(S) >= type_frequency_ratio):
                types.append(k)

    elif(type_filter_method == "top"):
        # https://careerkarma.com/blog/python-sort-a-dictionary-by-value/
        sort = sorted(types_dict.items(), key=lambda x: x[1], reverse=True)
        for i in range(0,type_top_n):
            types.append(sort[i][0])

    elif(type_filter_method == "top_percent"):
        # https://careerkarma.com/blog/python-sort-a-dictionary-by-value/
        sort = sorted(types_dict.items(), key=lambda x: x[1], reverse=True)
        total = len(types_temp)
        cutoff = int(total * (type_top_n_percent/100))
        for i in range(0,cutoff):
            types.append(sort[i][0])

    elif(type_filter_method == "top_variable"):
        df = pd.read_csv(working_path + "/most_variable_types.tsv",sep="\t",header=None)
        top_types = df[0][0:type_top_n_variable]
        for t in top_types:
            types.append(str(t))

    elif(type_filter_method == "taxonomy"):
        df = pd.read_csv(working_path + "/" + taxonomy_filename,sep="\t",header=None)
        for i,r in df.iterrows():
            original = r[0]
            mapping = r[1]

            taxonomy[original] = mapping

            if(mapping not in types):
                types.append(mapping)

    elif(type_filter_method == "none"):
        for k,v in types_dict.items():
            types.append(k)

    else:
        print("Invalid type filter method.")

    # Call taxonomy function

    if(type_filter_method == "taxonomy"):
        S = apply_taxonomy(S,taxonomy)

    return(S,types)