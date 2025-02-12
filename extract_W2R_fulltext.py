'''
This script is to extract W2R from full-text of review papers
Currently supports
- models: llama3.1, gpt-4o-mini
- method: full, chunk
- style: json, code1 (no code2)
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
import tqdm
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

from extract_W2R_compare import *


def split_into_chunks(text, chunk_size):
    length_function = len

    # The default list of split characters is [\n\n, \n, " ", ""]
    # Tries to split on them in order until the chunks are small enough
    # Keep paragraphs, sentences, words together as long as possible
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "."],
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=length_function,
    )
    splits = splitter.split_text(text)

    return splits


def evaluate_relatedness(text, model):
    system_prompt = '''
            Given a text, determine whether it contains waste-to-resource transformation that specifies what is the waste,
            what is the process, and what is the resource. 
            You should only answer with Yes or No. Do not explain yourself.
            '''
    user_prompt = '''
            text: {}
            result: 
            '''.format(text)
    
    response_string = prompt_llm(model, system_prompt, user_prompt)

    return response_string


def extract_fulltext_chunk(content, args):
    result_list = []
    chunk_id_list = []
    chunks = split_into_chunks(content, args.chunk_size)

    for i, chunk in enumerate(tqdm.tqdm(chunks)):
        relatedness_response = evaluate_relatedness(text=chunk, model=args.model)
        if "yes" in relatedness_response.lower():

            # extract and postprocess
            if args.style == "json":
                coded_chunk = chunk   # no need to wrap
                result_str = extract(coded_chunk, args)
                result_json = extract_postprocessing(result_str)   # a single json object

            elif args.style == "code":
                # wrap abstract to code style
                coded_chunk = wrap_abstract_to_codestyle(chunk)
                result_str = extract(coded_chunk, args)
                result_json = extract_postprocessing_codestyle(result_str)

            else:
                raise ValueError("Wrong style!")
            
            # add to result_list
            result_list.append(result_json)
            chunk_id_list.append(i)

            logging.info("Chunk: {}".format(chunk))
            logging.info("Extracted: {}".format(result_json))

    logging.info(f"Total {len(chunks)} chunks")
    logging.info(f"Total {len(result_list)} chunks has relatedness")

    return result_list, chunk_id_list


def process_fulltext(args):
    df = pd.read_csv("scopus_waste2resource.csv")

    # if start_index is not None, only process the rows from start_index to start_index+num
    if args.start_index is not None:
        df = df.iloc[args.start_index:]
    
    # Filter for review papers
    review_papers = df[(df['Document Type'] == 'Review')]

    if args.num > 0:
        review_papers = review_papers[:args.num]
    
    result_json_file = os.path.join(args.save_dir, "w2r_fulltext_results.json")
    result_invalid_file = os.path.join(args.save_dir, "w2r_fulltext_invalid.txt")
    result_invalid_doi_file = os.path.join(args.save_dir, "w2r_fulltext_invalid_doi.txt")

    result_json_list_final = []
    result_invalid_str = ""
    result_invalid_doi_list = []

    total_time = 0
    fulltext_available = 0

    # create subfolder to store results for each review paper
    each_paper_save_dir = os.path.join(args.save_dir, "review_papers")
    check_make_dir(each_paper_save_dir)

    for i, row in review_papers.iterrows():
        doi = str(row['DOI'])   # if empty, doi = nan, then need to convert to string
        filename = os.path.join("full_text_papers_processed", doi.replace('/', '_') + '.txt')

        print(f"Processing row {i}: {filename}")
        logging.info(f"Processing row {i}: {filename}")

        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
            fulltext_available += 1
            print(f"File found, processing...")
            logging.info(f"File found, processing...")
            
            start_time = time.time()

            # extract from fulltext using chunking method
            result_json_list, chunk_id_list = extract_fulltext_chunk(content, args)

            # check if extracted successfully
            if len(result_json_list) == 0:
                # result_invalid_str += result + "\n\n----------\n\n"
                result_invalid_doi_list.append(doi)
                result_invalid_json = {"waste": [], "transforming_process": [], "transformed_resource": [],
                                       "reference": doi}
                result_json_list.append(result_invalid_json)
                logging.info("No valid W2R information found")

            else:
                logging.info("Converted {} w2r to json".format(len(result_json_list)))
                result_json_list_i = []
                for result_json_item, chunk_id in zip(result_json_list, chunk_id_list):
                    result_json_item["reference"] = "{}_chunk{}".format(doi, chunk_id)
                    result_json_list_i.append(result_json_item)
                    result_json_list_final.append(result_json_item)

                    # save results for each fulltext
                    result_json_file_i = os.path.join(each_paper_save_dir, "w2r_fulltext_results_{}.json".format(i))
                    with open(result_json_file_i, 'w') as file:
                        json.dump(result_json_list_i, file, indent=4)

            # logging
            time_i = time.time() - start_time
            total_time += time_i
            
            logging.info(f"Processed {doi}, time in seconds: {time_i}")

    # save final results
    with open(result_json_file, 'w') as file:
        json.dump(result_json_list_final, file, indent=4)

    try:
        with open(result_invalid_file, 'w') as file:
            file.write(result_invalid_str)
    except:
        with open(result_invalid_file, 'w', encoding='utf-8') as file:
            file.write(result_invalid_str)

    with open(result_invalid_doi_file, 'w') as file:
        for invalid_doi in result_invalid_doi_list:
            file.write(invalid_doi + '\n')

    logging.info("-------------"*5)
    logging.info(f"Fulltext available: {fulltext_available} / {len(review_papers)}")
    logging.info(f"Successfully extracted: {len(result_json_list_final)}")
    logging.info(f"Total time: {total_time}, averaged time: {total_time/len(result_json_list_final) if result_json_list_final else 0} seconds")


def main(args):
    process_fulltext(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=50, help='How many review papers to process (including fulltext unvailable papers). Set -1 to process all.')
    parser.add_argument('--start_index', type=int, help='Start index for processing in the orignial csv file')
    parser.add_argument('--model', type=str, default='llama3.1',
                        choices=["llama3", "llama3.1", "gpt-3.5-turbo", "gpt-4o-mini"],
                        help='Model to extract W2R.')
    parser.add_argument('--model_relatedness', type=str, default='llama3.1',
                        choices=["llama3", "llama3.1", "gpt-3.5-turbo", "gpt-4o-mini"],
                        help='Model to evaluate relatedness.')
    parser.add_argument('--prompt', type=str, default='zero', choices=["zero", "few"],
                        help='Prompt strategy: zero/few')
    parser.add_argument('--style', type=str, default='code', choices=['json', 'code'],
                        help='Style of the extraction: json_multi or code')
    parser.add_argument('--temperature', type=float, default=1, help='temperature setting for the llm')
    parser.add_argument('--chunk_size', type=int, default=1000, help='chunk size for the text')
    parser.add_argument('--save_dir', type=str, help='root directory to save results.')
    parser.add_argument('--save_dir_rewrite', action='store_true',
                        help='whether to rewrite result files or increment folder name')
    parser.add_argument('--shot_k', type=int, default=2, help='number of examples for few-shot')
    parser.add_argument('--shot_ids', type=int, nargs='+', default=[], help='ids of shot examples')
    args = parser.parse_args()

    # if save_dir is not provided, use the default format: save_dir/{model}_{prompt}_{style}_{method}
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join("result_all", "fulltext", args.model + "_" + args.prompt + "_" + args.style + "_" + args.method)
    args.save_dir = check_make_dir(save_dir, exist_ok=args.save_dir_rewrite)  

    # set logging file
    log_file = os.path.join(args.save_dir, "extraction_log.log")
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(message)s')

    # Log each argument and its value
    for arg, value in vars(args).items():
        logging.info(f'Argument {arg}: {value}')
        print(f'{arg}: {value}')
    logging.info("-------------"*5)

    main(args)

    print("Result saved in {}".format(args.save_dir))
