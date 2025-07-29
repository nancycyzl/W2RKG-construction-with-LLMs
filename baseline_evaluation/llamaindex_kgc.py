'''
If add relation properites, some errors will occur.
Suspect its the LlamaIndex bug with OpenAI API call.
So currently there are no processes extracted.
'''

from llama_index.core import Document, KnowledgeGraphIndex, PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor, SimpleLLMPathExtractor
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
import argparse
import pandas as pd
import os
import json

from typing import Literal

def frame_few_shot_examples(text, w2r_dict, set_relation_prop):
    # (
    # EntityNode(label="WASTE", name=<waste>),
    # Relation(label="TRANSFORMED_INTO", source_id=<waste>, target_id=<resource>%s),
    # EntityNode(label="RESOURCE", name=<resource>)
    # )

    process_str = ', properties={"transforming_process": %s}' % (w2r_dict["transforming_process"]) if set_relation_prop else ""

    example = """
TEXT:
{text}

Triples:
(
    EntityNode(label="WASTE", name=%s),
    Relation(label="TRANSFORMED_INTO", source_id=%s, target_id=%s%s),
    EntityNode(label="RESOURCE", name=%s)
)""" % (w2r_dict["waste"], w2r_dict["waste"], w2r_dict["transformed_resource"], process_str, w2r_dict["transformed_resource"])

    return example

def create_prompt_with_schema(set_relation_prop=False, few_shot=0):

    # basic prompt
    prompt = """
You are an information extraction agent.
Extract a waste-to-resource knowledge graph from the given text, including waste, resource, and tranforming process.
Please follow this schma for each triple:
(
    EntityNode(label="WASTE", name=<waste>),
    Relation(label="TRANSFORMED_INTO", source_id=<waste>, target_id=<resource>%s),
    EntityNode(label="RESOURCE", name=<resource>)
)
""" % (', properties={"transforming_process": <process>}' if set_relation_prop else "")
    
    if few_shot > 0:
        # load few shot examples
        import sys, pathlib
        sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
        from prompts_repo import get_abstract_list

        abstract_list, w2r_list = get_abstract_list()
        for abstract, w2r_dict in zip(abstract_list[:few_shot], w2r_list[:few_shot]):
            example_string = frame_few_shot_examples(abstract, w2r_dict, set_relation_prop)
            prompt += "\n\n" + example_string

    # final prompt
    prompt += "\n\n" + """
Text:
{text}

Triples:
"""

    return prompt


def parse_triples(triples, reference):
    w2r_dict = {
        "waste": [],
        "transforming_process": [],
        "transformed_resource": [],
        "reference": [reference]
    }

    for waste_node, relation, resource_node in triples:
        waste = waste_node.name
        resource = resource_node.name
        process = relation.properties.get("transforming_process", None)

        w2r_dict["waste"].append(waste)
        w2r_dict["transforming_process"].append(process)
        w2r_dict["transformed_resource"].append(resource)

    
    return w2r_dict

def process_data(abstract_list, reference_list, args):

    # define llm
    llm = OpenAI(temperature=0, model=args.model)
    Settings.llm = llm
    Settings.chunk_size = 1024

    # define propmt
    prompt = create_prompt_with_schema(args.set_relation_prop, args.few_shot)

    # define schema
    entities = Literal["Waste", "Resource"] 
    relations = Literal["TRANSFORMED_INTO"]

    relation_props = None
    if args.set_relation_prop:
        relation_props = ["transforming_process"]

    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,
        extract_prompt=prompt,
        possible_entities=entities,
        possible_relations=relations,
        possible_relation_props=relation_props,
        max_triplets_per_chunk=20,
        strict=False,
    )


    # kg_extractor = SimpleLLMPathExtractor(
    #     llm=llm,
    #     extract_prompt=prompt,
    #     max_paths_per_chunk=20,
    # )

    # initialize result list
    result_list = []

    for i, (abstract, reference) in enumerate(zip(abstract_list, reference_list)):

        # # initialize graph store
        # graph_db = SimpleGraphStore()

        # prepare documents
        print(f"Processing reference: {reference} (index: {i+args.start_idx})")
        documents = [Document(text=abstract)]

        # kg index
        kg_index = PropertyGraphIndex.from_documents(
            documents,
            kg_extractors=[kg_extractor]
        )

        ## -----------------------------------------------------------------

        # save graph in html
        kg_index.property_graph_store.save_networkx_graph(name=f"{args.output_dir}/kg.html")

        # get graph store
        graph_store = kg_index.property_graph_store
        triples = graph_store.get_triplets(relation_names=["TRANSFORMED_INTO", "Transformed_into"])
        # print(f"Triples from graph_store.get_triplets(): {triples}")

        # parse triples
        w2r_dict = parse_triples(triples, reference)
        print(f"W2R_dict: {w2r_dict}")

        # remove duplicate
        try:
            for key in w2r_dict:
                seen = set()
                uniq = []
                for v in w2r_dict[key]:
                    if v not in seen:
                        uniq.append(v)
                        seen.add(v)
                w2r_dict[key] = uniq
        except Exception as e:
            print(f"Error in removing duplicates for reference: {reference} (index: {i+args.start_idx}): {e}")

        # append to final result list
        result_list.append(w2r_dict)
    

    return result_list


def main(args):
     
    df = pd.read_csv("scopus_waste2resource.csv")
    abstract_list = df["Abstract"].tolist()
    doi_list = df["DOI"].tolist()

    if args.num:
        abstract_list = abstract_list[args.start_idx:args.start_idx+args.num]
        doi_list = doi_list[args.start_idx:args.start_idx+args.num]

    result_list = process_data(abstract_list, doi_list, args)

    # save refile
    save_file = os.path.join(args.output_dir, f"llamaindex_result_{args.model}.json")
    with open(save_file, "w") as f:
        json.dump(result_list, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt-4o-mini", "gpt-4o"], default="gpt-4o-mini")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="baseline_evaluation")
    parser.add_argument("--set_relation_prop", action="store_true", help="Set relation property")
    parser.add_argument("--few_shot", type=int, default=0)
    args = parser.parse_args()

    # create output directory if it doesn't exist
    save_dir = os.path.join(args.output_dir, f"llamaindex_result_{args.model}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.output_dir = save_dir

    main(args)