from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.documents import Document
import pandas as pd
import argparse
import getpass
import asyncio
import json
import os

def verify_api_key():
    """Verify that the OpenAI API key is valid by making a test request"""
    try:
        # Create a test ChatOpenAI instance
        chat = ChatOpenAI()
        # Try to generate a simple response
        response = chat.invoke("Say 'API key is valid'")
        print("✓ API key verified successfully!")
        return True
    except Exception as e:
        print(f"✗ Error verifying API key: {str(e)}")
        return False
    

def frame_few_shot_examples(text, w2r_dict, set_relation_prop):
    '''
    Return a dict for one text.
    '''
    process_str = ',\n      "properties": {{\n        "transforming_process": %s\n      }}' % (w2r_dict["transforming_process"]) if set_relation_prop else ""
    example = """
{{
    "nodes": [
        {{"id": %s, "type": "Waste"}},
        {{"id": %s, "type": "Resource"}}
    ],
    "relationships": [
    {{
        "source": %s,
        "target": %s,
        "type": "TRANSFORMED_INTO"%s
    }}
    ]
}}""" % (w2r_dict["waste"], w2r_dict["transformed_resource"], w2r_dict["waste"], w2r_dict["transformed_resource"], process_str)

    return (f"TEXT: {text}", example)
    
def create_prompt_with_schema(set_relation_prop=False, few_shot=0):

    # basic prompt (set instrucitons on schema)
    prompt_basic = """
You are an information extraction agent. Extract a waste-to-resource knowledge graph from the given text.

Please follow this schema:
{{
  "nodes": [
    {{"id": <string>, "type": "Waste"}},
    {{"id": <string>, "type": "Resource"}},
    ...
  ],
  "relationships": [
    {{
      "source": <string>,
      "target": <string>,
      "type": "TRANSFORMED_INTO"%s
    }},
    ...
  ]
}}
""" % (
        ',\n      "properties": {{\n        "transforming_process": <waste-to-resource transforming process>\n      }}'
        if set_relation_prop
        else ""
    )

    msgs = [("system", prompt_basic)]

    if few_shot > 0:
        import sys, pathlib
        sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
        from prompts_repo import get_abstract_list

        abstract_list, w2r_list = get_abstract_list()
        for abstract, w2r_dict in zip(abstract_list[:few_shot], w2r_list[:few_shot]):
            human_text, ai_text = frame_few_shot_examples(abstract, w2r_dict, set_relation_prop)
            msgs.append(("human", human_text))
            msgs.append(("ai", ai_text))
        
    # user's real input
    msgs.append(("human", "TEXT: {input}"))

    return ChatPromptTemplate.from_messages(msgs)
        

async def process_data(abstract_list, args):
    # set LLM
    llm = ChatOpenAI(temperature=0, model_name=args.model)

    # create prompt
    prompt = create_prompt_with_schema(args.set_relation_prop, args.few_shot)

    relationship_properties=None
    if args.set_relation_prop:
        relationship_properties=['transforming_process']

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        prompt=prompt,
        allowed_nodes=["Waste", "Resource"],
        allowed_relationships=[("Waste", "TRANSFORMED_INTO", "Resource")],
        relationship_properties=relationship_properties
    )

    docs = [Document(page_content=abstract) for abstract in abstract_list]

    print("Processing all data...")
    graph_docs = await llm_transformer.aconvert_to_graph_documents(docs)

    return graph_docs


def parse_graph_docs(graph_doc, reference):
    '''
    Parse one graph document to a dict.
    '''
    # print("Graph_doc: ", graph_doc)

    w2r_dict = {
        "waste": [],
        "transforming_process": [],
        "transformed_resource": [],
        "reference": []
    }

    for rel in graph_doc.relationships:
        if rel.type == "TRANSFORMED_INTO":
            waste_name = rel.source.id
            resource_name = rel.target.id
            process = rel.properties.get("transforming_process")

            w2r_dict["waste"].append(waste_name)
            w2r_dict["transformed_resource"].append(resource_name)
            if process:
                w2r_dict["transforming_process"].append(process)

            w2r_dict["reference"].append(reference)

    for key in w2r_dict:
        seen = set()
        uniq = []
        for v in w2r_dict[key]:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        w2r_dict[key] = uniq

    # print("W2R_dict: ", w2r_dict)

    return w2r_dict
                


def main(args):

    # Check for API key in environment first
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    if not verify_api_key():
        print("Please check your API key and try again.")
        exit(1)
        
    df = pd.read_csv("scopus_waste2resource.csv")
    abstract_list = df["Abstract"].tolist()
    doi_list = df["DOI"].tolist()

    if args.num:
        abstract_list = abstract_list[args.start_idx:args.start_idx+args.num]
        doi_list = doi_list[args.start_idx:args.start_idx+args.num]

    result_list = []

    graph_docs = asyncio.run(process_data(abstract_list, args))
    for graph_doc, reference in zip(graph_docs, doi_list):
        w2r_dict = parse_graph_docs(graph_doc, reference)
        result_list.append(w2r_dict)
    
    save_file = os.path.join(args.output_dir, f"langchain_result_{args.model}.json")
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
    save_dir = os.path.join(args.output_dir, f"langchain_result_{args.model}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.output_dir = save_dir

    main(args)