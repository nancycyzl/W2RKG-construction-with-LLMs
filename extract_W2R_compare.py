'''
Extract from paper ABSTRACT using scopus_waste2resource.csv
Usage: python extract_W2R_compare.py --num 50 --model gpt-4o-mini --prompt zero --save_dir experiment_model_compare
Parameters:
-- model: llama3.1, gpt-4o-mini
-- prompt: zero, few
-- style: json, code
   1. json: output the result in json format
   2. code: output the result in code format (eg. w2r["waste"]=["waste1", "waste2"])
'''

import ollama
import time
import json
import pandas as pd
import argparse
from openai import OpenAI
import re
import os
import ast
from utils import *
import logging
from prompts_repo import get_few_shot_examples, get_system_prompt_basic


def extract(abstract, args):
    # model: llama3.1, gpt4o
    # prompt: zero, few, cot
    # style: json, code, code2

    # configure system prompt
    system_prompt = get_system_prompt_basic(style=args.style)

    if args.prompt == "zero":
        pass

    elif args.prompt == "few":
        examples = get_few_shot_examples(style=args.style, k=args.shot_k, shot_ids=args.shot_ids)
        system_prompt = system_prompt + '\n\nHere are some examples:\n\n' + examples

    else:
        print("Wrong prompting strategy!! Using default zero-shot prompting.")

    # configure user prompt
    if args.style == "json":
        user_prompt = "Paragraph: " + abstract + "\n Result: "

    elif args.style == "code":
        user_prompt = '''w2r = {{"waste": [], "transforming_process": [], "transformed_resource": []}}
                         text = "{}"
                         # now complete the code by extracting waste, transforming_process, transformed_resource. Show me the added code only.
                         '''.format(abstract)
    else:
        raise ValueError("Wrong style!")

    response_string = prompt_llm(args.model, system_prompt, user_prompt)

    return response_string


def extract_postprocessing(result):
    # Extract the JSON string from the text (a single JSON object)
    json_match = re.search(r'\{.*?\}', result, re.DOTALL)
    if json_match:
        json_content = json_match.group()
        try:
            # Convert the JSON string to a JSON object
            # json_data = json.loads(json_content)    # DO NOT USE THIS: valid json format use double quotes
            json_data = ast.literal_eval(json_content)    # USE THIS: convert string to dict, single or double quotes are both ok
            # Ensure the JSON object includes the required keys
            required_keys = ["waste", "transforming_process", "transformed_resource"]
            for key in required_keys:
                if key not in json_data:
                    json_data[key] = []
            logging.info("Conversion succeeds")
            return json_data
        except:
            # If conversion fails, create a JSON with specified keys and empty lists
            logging.info("Conversion fails")
            json_data = {
                "waste": [],
                "transforming_process": [],
                "transformed_resource": []
            }
            return json_data
    else:
        # If no JSON string is found, return a JSON with specified keys and empty lists
        logging.info("No JSON string found")
        json_data = {
            "waste": [],
            "transforming_process": [],
            "transformed_resource": []
        }
        return json_data
    

def extract_postprocessing_codestyle(code_string):
    w2r = {"waste": [], "transforming_process": [], "transformed_resource": []}
    try:
        # Regular expression pattern to match w2r assignments
        pattern = r'w2r\["(?P<key>\w+)"\]\s*=\s*(?P<value>\[.*?\])'
        # Find all matches in the code string
        matches = re.finditer(pattern, code_string, re.DOTALL)
        for match in matches:
            key = match.group('key')
            value_str = match.group('value')
            try:
                # Safely evaluate the list using ast.literal_eval
                value = ast.literal_eval(value_str)
                if isinstance(value, list):
                    w2r[key] = value
                else:
                    print(f"Warning: The value for '{key}' is not a list.")
            except Exception as e:
                print(f"Error evaluating the list for '{key}': {e}")
    except Exception as e:
        print("An error occurred while parsing the code string:", e)
    return w2r   


def wrap_abstract_to_codestyle(abstract):
    code_style_abstract = '''
        w2r = {{"waste": [], "transforming_process": [], "transformed_resource": []}}
        text = "{}"
        '''.format(abstract)
    return code_style_abstract


def process_abstracts(args):

    # num, model, prompt, save_dir, shot_length, shot_k

    result_json_file = os.path.join(args.save_dir, "w2r_results.json")
    result_invalid_file = os.path.join(args.save_dir, "w2r_invalid.txt")
    result_invalid_doi_file = os.path.join(args.save_dir, "w2r_invalid_doi.txt")

    df = pd.read_csv("scopus_waste2resource.csv")
    abstract_list = df["Abstract"].tolist()
    doi_list = df["DOI"].tolist()

    if args.start_index:
        abstract_list = abstract_list[args.start_index:]
        doi_list = doi_list[args.start_index:]

    if args.num > 0:
        abstract_list = abstract_list[:args.num]
        doi_list = doi_list[:args.num]

    result_json_list = []
    result_invalid_str = ""
    result_invalid_doi_list = []

    total_time = 0

    for i, abstract in enumerate(abstract_list):
        start_time = time.time()

        if args.style == "json":
            coded_abstract = abstract   # no need to wrap
            result_str = extract(coded_abstract, args)
            result_json = extract_postprocessing(result_str)

        elif args.style == "code":
            # wrap abstract to code style
            coded_abstract = wrap_abstract_to_codestyle(abstract)
            result_str = extract(coded_abstract, args)
            result_json = extract_postprocessing_codestyle(result_str)

        else:
            raise ValueError("Wrong style!")
        
        # add the referece and append
        result_json["reference"] = doi_list[i]
        result_json_list.append(result_json)

        # check if the extracted information is valid
        if len(result_json["waste"]) == 0 and len(result_json["transforming_process"]) == 0 and len(result_json["transformed_resource"]) == 0:
            result_invalid_str += result_str + "\n\n----------\n\n"
            result_invalid_doi_list.append(doi_list[i])

        logging.info("----------------------------------")
        logging.info("Processing abstract {}".format(i))
        logging.info("Coded abstract: \n{}".format(coded_abstract))
        logging.info("LLM response: \n{}".format(result_str))
        logging.info("Extracted w2r: \n{}".format(result_json))

        end_time = time.time()
        total_time += end_time - start_time
        logging.info("Time taken: {:.2f} seconds".format(end_time - start_time))
        print("Processing abstract {}, time taken: {:.2f} seconds".format(i, end_time - start_time))
        # end for loop

        # save results each 10 abstracts    
        if i % 10 == 0:
            with open(result_json_file, 'w') as file:
                json.dump(result_json_list, file, indent=4)

    # save the final result
    with open(result_json_file, 'w') as file:
        json.dump(result_json_list, file, indent=4)

    try:
        with open(result_invalid_file, 'w') as file:
            file.write(result_invalid_str)
    except:
        with open(result_invalid_file, 'w', encoding='utf-8') as file:
            file.write(result_invalid_str)

    with open(result_invalid_doi_file, 'w') as file:
        for invalid_doi in result_invalid_doi_list:   # if empty, doi = nan, then need to convert to string
            file.write(str(invalid_doi) + '\n')

    logging.info("-------------"*5)
    logging.info("Successfully extracted: {} / {}".format(len(result_json_list), len(abstract_list)))
    logging.info("Total time: {}, averaged time: {} seconds".format(total_time, total_time/len(result_json_list)))


def main(args):
    process_abstracts(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=50, help='How many abstracts to process. Set -1 to process all.')
    parser.add_argument('--start_index', type=int, help='Start index in the original csv file.')
    parser.add_argument('--model', type=str, default='llama3.1',
                        choices=["llama3", "llama3.1", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                        help='Model to use, llama or gpt series.')
    parser.add_argument('--prompt', type=str, default='zero', choices=["zero", "few"],
                        help='Prompt strategy: zero/few')
    parser.add_argument('--style', type=str, default='json', choices=['json', 'code'],
                        help='Style of the extraction: json or code')
    parser.add_argument('--temperature', type=float, default=1, help='temperature setting for the llm')
    parser.add_argument('--save_dir', type=str, default='result', help='root directory to save results.')
    parser.add_argument('--save_dir_rewrite', action='store_true',
                        help='whether to rewrite result files or increment folder name')
    parser.add_argument('--shot_k', type=int, default=2, help='number of examples for few-shot')
    parser.add_argument('--shot_ids', type=int, nargs='+', default=[], help='ids of shot examples')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat the experiment')
    args = parser.parse_args()

    if args.prompt == "zero":
        # create base folder: {model}_{prompt}_{style}
        base_folder = os.path.join(args.save_dir, args.model + "_" + args.prompt + "_" + args.style)
    else: # args.prompt == "few"
        # create base folder: {model}_{prompt}_{style}_k{shot_k}
        base_folder = os.path.join(args.save_dir, args.model + "_" + args.prompt + "_" + args.style + "_k" + str(args.shot_k))
    base_folder = check_make_dir(base_folder, exist_ok=args.save_dir_rewrite)

    for i in range(args.repeat):
        print("Running experiment {}/{}".format(i+1, args.repeat))
        logging.info("Running experiment {}/{}".format(i+1, args.repeat))
        # create subfolder in the save_dir: run_{i}   
        save_dir = os.path.join(base_folder, "run_{}".format(i))
        args.save_dir = check_make_dir(save_dir, exist_ok=args.save_dir_rewrite)

        # set logging file
        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()
        log_file = os.path.join(args.save_dir, "extraction_log.log")
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(message)s')

        # Log each argument and its value
        for arg, value in vars(args).items():
            logging.info(f'Argument {arg}: {value}')
            print(f'{arg}: {value}')
        logging.info("-------------"*5)

        main(args)