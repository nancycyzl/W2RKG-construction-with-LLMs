'''
Input file should be one ground-truth file + one result file, all as json files
Output files:
1. prediction_resolution.csv: match pr entities to the gt entities
2. metrics_intermediate.json: record TP/FP/FN
3. metrics.csv: record micro/macro precision, recall, f1, and Jaccard
'''

# API key is saved using: setx OPENAI_API_KEY "key"

from openai import OpenAI
import numpy as np
import argparse
import logging
import ollama
import os
import re
from utils import *
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


def clear_gpu_memory():
    torch.cuda.empty_cache()


def match_name(gt_list, pr_item, model="llama3.1"):
    system_prompt = "You are an evaluator and need to do some judgement."

    # convert to multiple choice question by adding A, B, C, ....
    choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
               "V", "W", "X", "Y", "Z"]
    gt_list = ["{}: {}".format(choice_index, item_name) for choice_index, item_name in zip(choices, gt_list)]
    user_prompt = '''Given a list and a test sample, is test sample semantically equivalent to any item in the list?
                     Answer with Yes or No. If the answer is Yes, indicate the index of the matched item directly.
                     Do not explain yourself. Here are some examples:
                     List: [\"A: date pits\", \"B: bio-wastes\", \"C: agro-wastes\"]. Sample: date pits.
                     Answer: Yes. A.
                     List: [\"A: peanut shells\", \"B: agro-wastes\"]. Sample: agricultural waste.
                     Answer: Yes. B.
                     List: [\"A: biochar\", \"B: hydrogen-rich gas\"]. Sample: hydrogen.
                     Answer: Yes. B.
                     List: [\"A: biochar\", \"B: hydrogen-rich gas\", \"C: fertilizer\", \"D: fuels\"]. Sample: liquid.
                     Answer: No.
                     List: {}. Sample: {}.
                     Answer:
                     '''.format(str(gt_list), pr_item)

    if "llama" in model.lower():
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ])
        return response['message']['content']

    elif "gpt" in model.lower():
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ])

        return response.choices[0].message.content


def resolution_llm(gt_list, one_pr, model):
    match_result = match_name(gt_list=gt_list, pr_item=one_pr, model=model).lower()
    logging.info("use LLM, match_result: {}".format(match_result))
    note = "llm match output: {}".format(match_result)
    if "yes" in match_result:
        # the return format should be : Yes. {index}.
        try:
            match_result_split = re.split(r'[,\.\'\"\(\)\[\]\{\}\?\!\s]+', match_result)  # idealy: ['yes', 'B', '']
            choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                       "S", "T", "U", "V", "W", "X", "Y", "Z"]
            matched_item = one_pr  # initialize matched item
            for x in match_result_split:
                if x.upper() in choices:
                    matched_item = gt_list[choices.index(x.upper())]
            resolved = matched_item
        except:
            resolved = one_pr

    elif "no" in match_result:
        resolved = one_pr

    else:
        logging.info("Error in match_name result: no 'yes' or 'no' found. Match result: {}".format(match_result))
        resolved = one_pr

    return resolved, note



def get_embedding(text):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True, device=device)
    
    # try:
    #     return model.encode(text, convert_to_tensor=True).cpu().numpy()
    
    # except:
    #     logging.warning("CUDA out of memory error encountered. Switching to CPU.")
    #     print("CUDA out of memory error encountered. Switching to CPU.")
    #     device = torch.device('cpu')
    #     model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True, device=device)
    #     return model.encode(text, convert_to_tensor=True).cpu().numpy()

    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True, device="cpu")
    embedding = model.encode(text, convert_to_tensor=True)
    embedding_np = embedding.cpu().numpy()
    return embedding_np


def resolution_emb(gt_list, one_pr, thre):
    # Call this function before evaluation
    clear_gpu_memory()

    logging.info("Matching using embeddings...")

    # getting embeddings
    pr_embedding = get_embedding(one_pr)
    gt_embeddings = get_embedding(gt_list)

    # Compute similarity scores
    similarity_scores = [1 - cosine(pr_embedding, gt_emb) for gt_emb in gt_embeddings]

    if len(similarity_scores) == 0:
        resolved = one_pr
        note = "No groundtruth"
    else:
        # Find the highest similarity score and its index
        max_score = max(similarity_scores)
        max_index = similarity_scores.index(max_score)

        # Check if the highest score is above the threshold
        if max_score > thre:
            resolved = gt_list[max_index]
            note = f"Matched with threshold {thre} (score: {max_score:.4f})"
        else:
            resolved = one_pr
            note = f"No match above threshold {thre} (highest score: {max_score:.4f})"

    return resolved, note


def evaluate_single(gt_list, pr_list, args):
    # gt, pr are both list of items
    logging.info("Using method: {}".format(args.method))
    if args.method == "llm":    
        logging.info("Using model: {}".format(args.model))
    elif args.method == "emb":
        logging.info("Using threshold: {}".format(args.thre))
    logging.info("Ground-truth list: {}".format(gt_list))

    resolved_pr_list = []
    note_list = []
    for one_pr in pr_list:
        logging.info("Resolving: {}".format(one_pr))
        # first step: if any term has exact string, then directly match
        if one_pr in gt_list:
            resolved_pr_list.append(one_pr)
            logging.info("directly matched: {}".format(one_pr))
            note_list.append("string exact match")
        # if string cannot match, then use semantic match using LLM or embedding
        else:
            if args.method == "llm":
                resolved_pr, note = resolution_llm(gt_list, one_pr, args.model)
            elif args.method == "emb":
                resolved_pr, note = resolution_emb(gt_list, one_pr, args.thre)
            resolved_pr_list.append(resolved_pr)
            note_list.append(note)  

    # save the resolution note to a csv file
    result_record = []  # list of list: ground-truth list string, prediction, resolved, label(TP/FP)
    gt_str = ", ".join(gt_list)
    for i, (orig_pr, res_pr) in enumerate(zip(pr_list, resolved_pr_list)):
        label = "TP" if res_pr in gt_list else "FP"
        result_record.append([gt_str, orig_pr, res_pr, label, note_list[i]])

    logging.info("Resolved prediction: {}".format(resolved_pr_list))
    logging.info("------------"*5)

    TP, FN, FP, jaccard = metrics_single(gt_list, resolved_pr_list)

    return TP, FN, FP, jaccard, result_record


def metrics_single(gt_list, pr_list):
    # pr_list is the resolved pr_list
    gt_set = set(gt_list)
    pr_set = set(pr_list)

    TP = len(gt_set & pr_set)                      # in both / intersection
    FN = len(gt_set - pr_set)                      # in ground-truth only
    FP = len(pr_set - gt_set)                      # in prediction only
    union = len(gt_set | pr_set)                   # union
    jaccard = TP/union if union != 0 else 0

    return TP, FN, FP, jaccard


def evaluate(args, pr_file):

    # gt_file, pr_file, model="llama3.1"
    TP_list = []
    FN_list = []
    FP_list = []
    jaccard_list = []
    metrics_list = []  # micro pre, micro rec, micro f1, macro pre, macro rec, macro f1, jaccard
    result_list = []  # list of list: i, waste/resource, ground-truth list string, prediction, resolved, label(TP/FP), note(string match or llm match)

    gt_json = read_json(args.gt_file)   # a list of json
    pr_json = read_json(pr_file)   # a list of json

    total_count = len(gt_json)
    invalid_count = 0  # the number of abstracts that have invalid structure

    for i, (gt, pr) in enumerate(zip(gt_json, pr_json)):   # for each ground-truth - result pair
        print("Evaluating {} / {}".format(i, len(gt_json)))
        logging.info("Evaluating {} / {}".format(i, len(gt_json)))
        expected_keys = {'waste', 'transforming_process', 'transformed_resource'}
        if not expected_keys.issubset(pr.keys()):
            invalid_count += 1
            continue
        waste_pr = pr["waste"]
        process_pr = pr["transforming_process"]
        resource_pr = pr["transformed_resource"]
        if len(waste_pr) != 0 or len(resource_pr) != 0 or len(process_pr) != 0:  # if all list are empty -> no valid json extracted
            waste_gt = remove_plural_to_lower(gt["waste"])           # all items to lower case and remove plural form first
            resource_gt = remove_plural_to_lower(gt["transformed_resource"])
            waste_pr = remove_plural_to_lower(waste_pr)
            resource_pr = remove_plural_to_lower(resource_pr)

            TP, FN, FP, jaccard, records = evaluate_single(waste_gt, waste_pr, args)
            TP_list.append(TP)
            FN_list.append(FN)
            FP_list.append(FP)
            jaccard_list.append(jaccard)
            # add i and waste/resource to the beginning of the records
            records = [[i, "waste"] + inner_list for inner_list in records]
            result_list.extend(records)

            TP, FN, FP, jaccard, records = evaluate_single(resource_gt, resource_pr, args)
            TP_list.append(TP)
            FN_list.append(FN)
            FP_list.append(FP)
            jaccard_list.append(jaccard)
            # add i and waste/resource to the beginning of the records
            records = [[i, "resource"] + inner_list for inner_list in records]
            result_list.extend(records)
        else:
            invalid_count += 1

    assert invalid_count + len(TP_list)/2 == total_count, "make sure each valid pair has two results, invalid count: {}, TP_list length: {}, total count: {}".format(invalid_count, len(TP_list), total_count)
    valid_count = total_count - invalid_count
    print("Total valid abstract: {} / {}".format(valid_count, total_count))
    logging.info("Total valid abstract: {} / {}".format(valid_count, total_count))

    metrics_intermediate_file = os.path.join(args.save_dir, "metrics_intermediate_{}.json".format(args.method))
    save_metrics_as_json(TP_list, FN_list, FP_list, jaccard_list, file_path=metrics_intermediate_file)
    print("TP, FN, FP, jaccard for each paper is saved at: ", metrics_intermediate_file)

    # save pr resolution
    pr_resolve_path = os.path.join(args.save_dir, "prediction_resolution_{}.csv".format(args.method))
    save_2d_list(result_list, headers=['i', 'type', 'ground-truth', 'prediction', 'resolved', 'label', 'note'],
                 filepath=pr_resolve_path)
    logging.info("Saved prediction result resolution to {}".format(pr_resolve_path))

    # micro: accumulate TP/FN/FP and then calculate precision/recall
    total_TP = sum(TP_list)
    total_FN = sum(FN_list)
    total_FP = sum(FP_list)
    precision_micro = (valid_count / total_count) * total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0  # weighted average
    recall_micro = (valid_count / total_count) * total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0  # weighted average
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
    print("Weighted micro precision: {}, recall: {}, f1: {}".format(precision_micro, recall_micro, f1_micro))
    logging.info("Weighted micro precision: {}, recall: {}, f1: {}".format(precision_micro, recall_micro, f1_micro))
    metrics_list.extend([precision_micro, recall_micro, f1_micro])

    # macro: for each paper, calculate precision/recall then take the average
    precision_list = []
    recall_list = []
    f1_list = []
    for tp, fn, fp in zip(TP_list, FN_list, FP_list):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    precision_macro = sum(precision_list) / (total_count*2)   # weighted average, total_count*2 because each valid pair has two results
    recall_macro = sum(recall_list) / (total_count*2)         # weighted average
    f1_macro = sum(f1_list) / (total_count*2)                 # weighted average
    print("Weighted macro precision: {}, recall: {}, f1: {}".format(precision_macro, recall_macro, f1_macro))
    logging.info("Weighted macro precision: {}, recall: {}, f1: {}".format(precision_macro, recall_macro, f1_macro))
    metrics_list.extend([precision_macro, recall_macro, f1_macro])

    # jaccard
    jaccard = sum(jaccard_list) / (total_count*2)   # weighted average, total_count*2 because each valid pair has two results
    metrics_list.append(jaccard)
    print("Averaged jaccard: ", jaccard)
    logging.info("Averaged jaccard: {}".format(jaccard))

    # save reulst to metrics.csv file
    save_metrics_df(metrics_list, mode='w', filepath=os.path.join(args.save_dir, "metrics.csv"))

    if args.metrics_update_file:
        save_metrics_df(metrics_list, mode='a', filepath=args.metrics_update_file, comment='{}_few_{}'.format(args.model, args.comment))

    return metrics_list


def main(args):
    if args.pr_folder:
        logging.info("Evaluating files in {}".format(args.pr_folder))

        base_save_dir = args.save_dir

        # check if metrics_update_file is given
        if not args.metrics_update_file:
            args.metrics_update_file = os.path.join(base_save_dir, "metrics_all_runs.csv")
            logging.info("No metrics update file is given, metrics will be saved to {}".format(args.metrics_update_file))

        # Loop through all subfolders in the specified directory: args.pr_folder/run_i/w2r_results.json
        metrics_all_runs = []
        for run_folder in os.listdir(args.pr_folder):
            if "run" in run_folder:
                logging.info("------------"*5)
                logging.info("Evaluating run: {}".format(run_folder))
                print("Evaluating run: {}".format(run_folder))
                w2r_path = os.path.join(args.pr_folder, run_folder, "w2r_results.json")
                # set some parameters for each run
                args.comment = run_folder
                args.save_dir = os.path.join(args.pr_folder, run_folder)   # args.save_dir changes for each run
                # evaluate each run
                metrics = evaluate(args, pr_file=w2r_path)
                metrics_all_runs.append(metrics)

        # calculate mean and std of metrics across all runs
        metrics_mean = np.mean(np.array(metrics_all_runs, dtype=float), axis=0)
        metrics_std = np.std(np.array(metrics_all_runs, dtype=float), axis=0, ddof=1)   # ddof=1 means sample standard deviation

        print("Mean metrics across all runs: {}".format(metrics_mean))
        print("Standard deviation of metrics across all runs: {}".format(metrics_std))

        logging.info("Mean metrics across all runs: {}".format(metrics_mean))
        logging.info("Standard deviation of metrics across all runs: {}".format(metrics_std))

        # save mean and std of metrics to metrics_summary.csv file
        metrics_summary = ["{:.4f}+-{:.4f}".format(mean, std) for mean, std in zip(metrics_mean, metrics_std)]  # Create a string for each metric
        metrics_summary_df = pd.DataFrame([metrics_summary], columns=['micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_f1', 'jaccard'])
        summary_file_path = os.path.join(base_save_dir, "metrics_summary.csv")
        metrics_summary_df.to_csv(summary_file_path, index=False)
        logging.info("Mean and standard deviation of metrics saved to {}".format(summary_file_path))

    else:
        logging.info("Evaluating file: {}".format(args.pr_file))
        metrics = evaluate(args, pr_file=args.pr_file)


if __name__ == '__main__':
    # # check whether API key is successfully retrieved
    # print(os.environ.get("OPENAI_API_KEY"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='llm', choices=['llm', 'emb'], help='Method to use for matching.')
    parser.add_argument('--model', type=str, default='llama3.1',
                        choices=["llama3", "llama3.1", "gpt-3.5-turbo", "gpt-4o-mini"],
                        help='Model to use.')
    parser.add_argument('--thre', type=float, default=0.8, help='Threshold for embedding similarity.')
    parser.add_argument('--gt_file', type=str, default='extraction_groundtruth.json', help='Ground truth file.')
    parser.add_argument('--pr_file', type=str, default=None, help='LLM result file in json.')
    parser.add_argument('--pr_folder', type=str, default=None, help='result folder containing all runs')
    parser.add_argument('--save_dir', type=str, default=None, help='directory to save results.')
    parser.add_argument('--metrics_update_file', type=str, default='', help='metrics path that will be updated')
    parser.add_argument('--comment', type=str, default='', help='additional comment to put in the metrics comment column')
    args = parser.parse_args()

    # check whether pr_file or pr_folder is given
    if not args.pr_file and not args.pr_folder:
        raise ValueError("Either --pr_file or --pr_folder must be provided.")

    # save folder manipulation, if no folder given, save in the pr_file's folder / pr_folder
    if args.save_dir:
        args.save_dir =check_make_dir(args.save_dir)
    else:
        args.save_dir = os.path.dirname(args.pr_file) if args.pr_file else args.pr_folder
    
    # set up logging
    log_file = os.path.join(args.save_dir, "extraction_evaluation_with_gt_result_{}.log".format(args.method))
    logging.basicConfig(
        filename=log_file,  # Name of the file where log messages are saved
        filemode='w',
        level=logging.INFO,
        format='%(message)s'
    )

    logging.info("Parameters:")
    for key, value in vars(args).items():
        logging.info(f"{key}={value}")

    # evaluate
    main(args)
    print("Log file is saved at: ", log_file)