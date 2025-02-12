import json
import logging
import os
import argparse
import ollama
import tqdm
import re
from openai import OpenAI
import json
import ast
import os
import sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from utils import *


def load_and_process_w2r_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    wastes = set()
    resources = set()
    w2r_triples = []

    for item in data:
        item_wastes = item.get('waste', [])
        item_resources = item.get('transformed_resource', [])
        wastes.update(item_wastes)
        resources.update(item_resources)

        item_process = item.get('transforming_process', [])
        item_process = ", ".join(item_process)   # convert list to string

        item_reference = item.get('reference', '')
        # if item_reference == np.nan:
        #     item_reference = ""

        for waste in item_wastes:
            for resource in item_resources:
                w2r_triples.append((waste, resource, item_process, item_reference))

    return list(wastes), list(resources), w2r_triples


def create_embeddings(text_list):
    print("Creating embeddings...")
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    embedding = model.encode(text_list, convert_to_tensor=True)
    embedding_np = embedding.cpu().numpy()
    print("Embeddings created")

    return embedding_np


def hac_clustering(embeddings, distance_threshold=0.2):
    # Calculate pairwise distance matrix
    dist_matrix = squareform(pdist(embeddings, metric='cosine'))   # cosine distance = 1 - cosine similarity
    # Perform Agglomerative Clustering using a distance threshold
    clustering = AgglomerativeClustering(affinity='precomputed', linkage='complete',
                                         distance_threshold=distance_threshold, n_clusters=None)
    labels = clustering.fit_predict(dist_matrix)
    return labels


def convert_list_to_string(name_list):
    return ', '.join(name_list)


def convert_string_to_list(name_string):
    # Replace newlines with commas to handle both cases
    name_string = name_string.replace('\n', ', ')
    # Split the string by commas and strip whitespace
    return [element.strip() for element in name_string.split(', ') if element.strip()]


def unify_names(name_list, name_type="waste", model='llama3.1'):
    vague_examples = "wastes, waste materials, by-products" if name_type == "waste" else "recycled resources, green materials"
    system_prompt = '''Given a list of {} names, provide unified names for them which aggregate similar concepts. Here are some rules:
                        1. If any items are semantically similar, aggregate them into one unified representative name.
                        2. Avoid using abbreviations in the unified names.
                        3. The unified name should be a specific {} concept, avoid using vague or general concepts like {}
                        4. Each unified name should refer to a single concept, avoid using "or" in the unified names.
                        5. Directly output the unified names without any additional information or explanation.
                        '''.format(name_type, name_type, vague_examples)
    if name_type == "waste":
        user_prompt = '''Here are some examples:
                        Names: incineration bottom ash, IBM, treated incineration bottom ash (IBM), untreated IBM
                        Unified name: incineration bottom ash

                        Names: crop residues, agricultural wastes, crop wastes, agri-wastes, fruit wastes, fruit processing wastes
                        Unified name: agricultural waste, fruit waste
                        
                        Based on the examples above, provide unified names for the following names:
                        Names: {}
                        Unified names:
                        '''.format(convert_list_to_string(name_list))
    elif name_type == "resource":
        user_prompt = '''Here are some examples:
                        Names: hydrogen, H2, hydrogen-rich gas
                        Unified name: hydrogen

                        Names: renewable energy, electricity, biogas, green electricity, biomethane
                        Unified name: electricity, biogas, biomethane
                        
                        Based on the examples above, provide unified names for the following names:
                        Names: {}
                        Unified names:
                        '''.format(convert_list_to_string(name_list))
    else:
        raise ("Name type {} is not supported!".format(name_type))
    
    response_string = prompt_llm(model, system_prompt, user_prompt)
    
    logging.info("Original names: {}".format(name_list))
    logging.info("Unified names: {}".format(response_string))

    return convert_string_to_list(response_string)


def get_cluster_and_unified_names(unique_wastes, unique_resources, waste_cluster_labels, resource_cluster_labels, args):
    #### get cluster elements
    waste_cluster_elements = {}
    for i, waste in enumerate(unique_wastes):
        cluster_label = waste_cluster_labels[i]
        if cluster_label not in waste_cluster_elements:
            waste_cluster_elements[cluster_label] = []
        waste_cluster_elements[cluster_label].append(waste)

    resource_cluster_elements = {}
    for i, resource in enumerate(unique_resources):
        cluster_label = resource_cluster_labels[i]
        if cluster_label not in resource_cluster_elements:
            resource_cluster_elements[cluster_label] = []
        resource_cluster_elements[cluster_label].append(resource)

    # sort the clusters by labels
    waste_cluster_elements = dict(sorted(waste_cluster_elements.items()))
    resource_cluster_elements = dict(sorted(resource_cluster_elements.items()))

    print("Waste clusters: {}".format(len(waste_cluster_elements.keys())))
    print("Resource clusters: {}".format(len(resource_cluster_elements.keys())))
    logging.info("Waste clusters: {}".format(len(waste_cluster_elements.keys())))
    logging.info("Resource clusters: {}".format(len(resource_cluster_elements.keys())))

    # Ensure the output directory exists
    os.makedirs(args.save_path, exist_ok=True)

    # Save waste_cluster_elements
    with open(os.path.join(args.save_path, "waste_cluster_elements.json"), 'w') as waste_file:
        json.dump(waste_cluster_elements, waste_file, indent=4)

    # Save resource_cluster_elements
    with open(os.path.join(args.save_path, "resource_cluster_elements.json"), 'w') as resource_file:
        json.dump(resource_cluster_elements, resource_file, indent=4)

    ###  get the cluster unified names
    waste_cluster_unified_names = {}
    num_waste_clusters = len(waste_cluster_elements.keys())
    for cluster_label, cluster_elements in waste_cluster_elements.items():
        print("unifying waste for cluster label {} / {}".format(cluster_label, num_waste_clusters))
        waste_unified_names = unify_names(cluster_elements, name_type="waste", model=args.model_unify_names)
        waste_cluster_unified_names[cluster_label] = waste_unified_names

        # Save waste_cluster_unified_names
        with open(os.path.join(args.save_path, "waste_cluster_unified_names.json"), 'w') as waste_unified_file:
            json.dump(waste_cluster_unified_names, waste_unified_file, indent=4)

    ## TEMP read cluster element
    with open("result_all/after_fusion_v2/thre09_complete/resource_cluster_elements.json") as file:
        resource_cluster_elements = json.load(file)

    resource_cluster_unified_names = {}
    num_resource_clusters = len(resource_cluster_elements.keys())
    for cluster_label, cluster_elements in resource_cluster_elements.items():
        if int(cluster_label) >= 4920:
            print("unifying resource for cluster label {} / {}".format(cluster_label, num_resource_clusters))
            resource_unified_names = unify_names(cluster_elements, name_type="resource", model=args.model_unify_names)
            resource_cluster_unified_names[cluster_label] = resource_unified_names
        
        # Save resource_cluster_unified_names
        with open(os.path.join(args.save_path, "resource_cluster_unified_names.json"), 'w') as resource_unified_file:
            json.dump(resource_cluster_unified_names, resource_unified_file, indent=4)

    return waste_cluster_elements, resource_cluster_elements, waste_cluster_unified_names, resource_cluster_unified_names


def load_cluster_info(path):
    
    waste_cluster_elements_file = os.path.join(path, "waste_cluster_elements.json")
    resource_cluster_elements_file = os.path.join(path, "resource_cluster_elements.json")
    waste_cluster_file = os.path.join(path, "waste_cluster_unified_names.json")
    resource_cluster_file = os.path.join(path, "resource_cluster_unified_names.json")

    with open(waste_cluster_elements_file, 'r') as waste_elements_file:
        waste_cluster_elements = json.load(waste_elements_file)

    with open(resource_cluster_elements_file, 'r') as resource_elements_file:
        resource_cluster_elements = json.load(resource_elements_file)

    with open(waste_cluster_file, 'r') as waste_file:
        waste_cluster_unified_names = json.load(waste_file)

    with open(resource_cluster_file, 'r') as resource_file:
        resource_cluster_unified_names = json.load(resource_file)

    return waste_cluster_elements, resource_cluster_elements, waste_cluster_unified_names, resource_cluster_unified_names


def get_cluster_labels(cluster_elements_dict, unique_elements):
    cluster_labels = []
    for element in unique_elements:
        for cluster_label, cluster_element in cluster_elements_dict.items():
            if element in cluster_element:
                cluster_labels.append(cluster_label)
                break

    assert len(cluster_labels) == len(unique_elements), "The length of cluster labels and unique elements should be the same!"
    return cluster_labels


def choose_one_unified_name(item, unified_names, model):
    embedding_item = model.encode(item, convert_to_tensor=True).cpu().numpy()
    embedding_list = model.encode(unified_names, convert_to_tensor=True).cpu().numpy()

    similarities = cosine_similarity(embedding_item.reshape(1, -1), embedding_list)
    most_similar_index = similarities.argmax()
    return unified_names[most_similar_index]
    

def fused_triples_post_processing(triples):
    # aggregate same waste and resource
    aggregated_triples = {}
    
    for triple in triples:
        waste = triple['waste']
        resource = triple['transformed_resource']
        key = (waste, resource)

        if key not in aggregated_triples:
            aggregated_triples[key] = {
                'waste': waste,
                'transforming_process': [triple['transforming_process']],
                'transformed_resource': resource,
                'reference': [triple['reference']]
            }
        else:
            if triple['transforming_process'] not in aggregated_triples[key]['transforming_process']:
                aggregated_triples[key]['transforming_process'].append(triple['transforming_process'])
            if triple['reference'] not in aggregated_triples[key]['reference']:
                aggregated_triples[key]['reference'].append(triple['reference'])

    # Convert aggregated_triples back to a list of dictionaries
    aggregated_triples = [
        {
            'waste': waste,
            'transforming_process': ', '.join(data['transforming_process']),
            'transformed_resource': resource,
            'reference': ', '.join(data['reference'])
        }
        for (waste, resource), data in aggregated_triples.items()
    ]

    return aggregated_triples


def main(args):
    
    # Load and process data
    unique_wastes, unique_resources, w2r_triples = load_and_process_w2r_data(args.input_file)   # list, list, list [(waste, resource, process, reference), ...]
    print(f"Number of unique wastes: {len(unique_wastes)}")
    print(f"Number of unique resources: {len(unique_resources)}")
    print(f"Number of w2r triples: {len(w2r_triples)}")
    logging.info(f"Number of unique wastes: {len(unique_wastes)}")
    logging.info(f"Number of unique resources: {len(unique_resources)}")
    logging.info(f"Number of w2r triples: {len(w2r_triples)}")

    # load cluster infor or perform HAC clustering and save the results
    if args.cluster_info_path:
        print(f"Loading waste and resource cluster unified names from {args.cluster_info_path}...")
        logging.info(f"Loading waste and resource cluster unified names from {args.cluster_info_path}...")
        waste_cluster_elements, resource_cluster_elements, waste_cluster_unified_names, resource_cluster_unified_names = load_cluster_info(args.cluster_info_path)  # 4 dicts
        logging.info("Total waste clusters: {}".format(len(waste_cluster_elements)))
        logging.info("Total waste unified names: {}".format(sum(len(waste_list) for waste_list in waste_cluster_unified_names.values())))
        logging.info("Total resource clusters: {}".format(len(resource_cluster_elements)))
        logging.info("Total resource unified names: {}".format(sum(len(resource_list) for resource_list in resource_cluster_unified_names.values())))
        print("Reconstructing waste and resource cluster labels...")
        waste_cluster_labels = get_cluster_labels(waste_cluster_elements, unique_wastes)
        resource_cluster_labels = get_cluster_labels(resource_cluster_elements, unique_resources)

    else:
        print("Processing HAC clustering and get cluster unified names, and save them to {}...".format(args.save_path))
        waste_embeddings = create_embeddings(unique_wastes)
        waste_cluster_labels = hac_clustering(waste_embeddings, distance_threshold=1-args.waste_threshold)
        waste_cluster_labels = waste_cluster_labels.tolist()  # Convert numpy.ndarray to a list of integers
        
        resource_embeddings = create_embeddings(unique_resources)
        resource_cluster_labels = hac_clustering(resource_embeddings, distance_threshold=1-args.resource_threshold)
        resource_cluster_labels = resource_cluster_labels.tolist()  # Convert numpy.ndarray to a list of integers

        waste_cluster_elements, resource_cluster_elements, waste_cluster_unified_names, resource_cluster_unified_names = get_cluster_and_unified_names(
            unique_wastes, unique_resources, waste_cluster_labels, resource_cluster_labels, args)   # 4 dicts

        print(f"Number of waste clusters: {len(np.unique(waste_cluster_labels))}")
        print(f"Number of resource clusters: {len(np.unique(resource_cluster_labels))}")
        logging.info(f"Number of waste clusters: {len(np.unique(waste_cluster_labels))}")
        logging.info(f"Number of resource clusters: {len(np.unique(resource_cluster_labels))}")

    
    # fuse triples at the ENTITY level
    print("Fusing triples...")
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    fused_triples = []
    for waste, resource, process, reference in w2r_triples:

        # convert names to unified names
        waste_cluster_label = waste_cluster_labels[unique_wastes.index(waste)]
        waste_unified_names = waste_cluster_unified_names[waste_cluster_label]   # may be multiple names
        waste_unified_name = waste_unified_names[0] if len(waste_unified_names) ==1 else choose_one_unified_name(waste, waste_unified_names, model)

        resource_cluster_label = resource_cluster_labels[unique_resources.index(resource)]
        resource_unified_names = resource_cluster_unified_names[resource_cluster_label]   # may be multiple names
        resource_unified_name = resource_unified_names[0] if len(resource_unified_names) ==1 else choose_one_unified_name(resource, resource_unified_names, model)

        # output the fused triple
        triple_json = {}
        triple_json["waste"] = waste_unified_name
        triple_json["transforming_process"] = process
        triple_json["transformed_resource"] = resource_unified_name
        triple_json["reference"] = reference
        fused_triples.append(triple_json)

        # logging.info("original triple: waste: {}, resource: {}, process: {}, reference: {}".format(waste, resource, process, reference))
        # logging.info("fused triple: {}".format(fused_triples))

    # save the fused triples
    with open(os.path.join(args.save_path, "fused_triples.json"), 'w') as file:
        json.dump(fused_triples, file, indent=4)
    logging.info("Number of fused triples: {}".format(len(fused_triples)))

    # aggregate same waste and resource and save
    fused_triples = fused_triples_post_processing(fused_triples)
    with open(os.path.join(args.save_path, "fused_triples_aggregated.json"), 'w') as file:
        json.dump(fused_triples, file, indent=4)
    logging.info("Number of fused triples after aggregation: {}".format(len(fused_triples)))
        

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fuse all triples into a KG')
    parser.add_argument('--input_file', type=str, default="result_all/before_fusion_v2/all_w2r_list.json", help='path for the w2r_results.json file')
    parser.add_argument('--save_path', type=str, default='results_all/after_fusion_v2', help='Path to save the results')
    parser.add_argument('--waste_threshold', type=float, default=0.8, help='Similarity threshold for waste')
    parser.add_argument('--resource_threshold', type=float, default=0.8, help='Similarity threshold for resource')
    parser.add_argument('--model_unify_names', type=str, default="llama3.1", choices=["llama3.1", "gpt-4o-mini"], help="llm for unify names")
    parser.add_argument('--model', type=str, default="llama3.1", choices=["llama3.1", "gpt-4o-mini"], help="llm for fuse triples")
    parser.add_argument('--cluster_info_path', type=str, help='path to load the cluster information dataset, if not specified, will process HAC clustering and save')
    parser.add_argument('--log_file', type=str, default='fusion_triples.log', help='Log file name')
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Set up logging and overwrite the log file
    logging.basicConfig(filename=os.path.join(args.save_path, args.log_file), level=logging.INFO, format='%(message)s', filemode='a')
    
    # Log the parameters set in argparse
    logging.info("Parameters:")
    for key, value in vars(args).items():
        logging.info(f"{key}={value}")
    
    main(args)
