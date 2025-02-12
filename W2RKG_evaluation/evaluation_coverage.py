'''
This script is to evaluate the coverage of W2RKG, and compare it with Maestri.

A. Agriculture, forestry and fishing
B. Mining and quarrying
C. Manufacturing
D. Electricity, gas, steam and air conditioning supply
E. Water supply; sewerage, waste management and remediation activities
F. Construction

G. Wholesale and retail trade; repair of motor vehicles and motorcycles
H. Transportation and storage
I. Accommodation and food service activities
J. Information and communication
K. Financial and insurance activities
L. Real estate activities
M. Professional, scientific and technical activities
N. Administrative and support service activities
O. Public administration and defence; compulsory social security
P. Education
Q. Human health and social work activities
R. Arts, entertainment and recreation
S. Other service activities
T. Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use
U. Activities of extraterritorial organisations and bodies

'''
import json
import os
import tqdm
import re
import argparse
import logging
import pandas as pd
import numpy as np
from utils import prompt_llm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def get_NACE_sector():
    # get the NACE standard sectors
    standard_path = "data/Standards.xlsx"
    nace_df = pd.read_excel(standard_path, sheet_name='NACE')
    nace_code_list = nace_df["NACE Rev. 2"].tolist()
    # print("Total NACE codes in all levels: ", len(nace_code_list))
    # print("Example of NACE codes: \n", nace_code_list[50:70])

    sector_dict = {}
    current_sector = None
    for item in nace_code_list:
        if len(str(item)) == 1:  # If the item is an alphabet, it's a new sector
            current_sector = item
            if current_sector not in sector_dict:
                sector_dict[current_sector] = []  # Initialize the sector with an empty list
        else:
            sector_dict[current_sector].append(str(item))

    # print("NACE sector: description")
    # for sector in sector_dict.keys():
    #     print(f"Sector ---{sector}----, type is {type(sector)}")
    #     description = nace_df[nace_df["NACE Rev. 2"] == sector]["Description"][0]
    #     print("Sector {}: {}".format(sector, description))

    return sector_dict


def get_sector_from_code(sector_dict, code):
    # check whether code is valid
    if "/" in code:
        codes = code.split("/")
        codes = [code.strip() for code in codes]
        code = codes[0]
    # find in the sector_dict
    for sector, values in sector_dict.items():
        if code in values:
            return sector
    return None


def get_sector_from_dict(sector_dict, item):
    for sector, values in sector_dict.items():
        if item in values:
            return sector
    return None

def save_confusion_matrix(confusion_matrix, save_folder, filename_wo_extension, color_map="coolwarm"):
    # save csv
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix,
        index=["A", "B", "C", "D", "E", "F", "other"],
        columns=["A", "B", "C", "D", "E", "F", "other"]
    )
    confusion_matrix_df.to_csv(os.path.join(save_folder, filename_wo_extension + ".csv"))

    # save plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=color_map, cbar=True,
                xticklabels=["A", "B", "C", "D", "E", "F", "other"],
                yticklabels=["A", "B", "C", "D", "E", "F", "other"],
                annot_kws={"size": 16})   # vmin=0, vmax=7500
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Receiving Sector", fontsize=16)
    plt.ylabel("Providing Sector", fontsize=16)
    plt.savefig(os.path.join(save_folder, filename_wo_extension + ".png"), dpi=300, bbox_inches='tight')


def assess_maestri(require_return=False, save_folder="compare_Maestri"):

    maestri_df = pd.read_excel("data/Maestri.xlsx", header=[0, 1, 2])  # Read first 3 rows as headers
    maestri_df.columns = ['_'.join(map(str, col)).strip() for col in maestri_df.columns]
    maestri_df.columns = ["id", "donor_name", "donor_business", "donor_NACE", "receiver_name", "receiver_business", "receiver_NACE",
                          "waste_description", "waste_EWC", "waste_CPA", "waste_CAS", "waste_EWC_hazardous",
                          "treatment_owner", "treatment description", "treatment_company", "treatment_business", "treatment_NACE",
                          "final_use", "source_quantities", "is_payment_due", "completion_level"]

    # # Display the first few rows
    # print(maestri_df.columns)

    # if final use is "raw material", change to "raw material for receiver_main_business"
    maestri_df.loc[maestri_df["final_use"] == "Raw material", "final_use"] = "Raw material for " + maestri_df["receiver_business"]

    # get the wastes
    waste_list = maestri_df['waste_description'].tolist()
    waste_unique = list(set(waste_list))
    print("Total wastes: {}, unique wastes: {}".format(len(waste_list), len(waste_unique)))

    # get the resources
    resource_list = maestri_df['final_use'].tolist()
    resource_unique = list(set(resource_list))
    print("Total resources: {}, unique resources: {}".format(len(resource_list), len(resource_unique)))

    # get sector dict
    nace_sector_dict = get_NACE_sector()

    # get provider NACE
    provider_list = maestri_df["donor_NACE"].tolist()
    provider_list = [str(nace) for nace in provider_list]
    provider_unique = list(set(provider_list))
    provider_2digit_list = [nace[:2] for nace in provider_list]
    provider_2digit_unique = list(set(provider_2digit_list))
    provider_sector_list = [get_sector_from_code(nace_sector_dict, nace) for nace in provider_list]
    # print("Example of provider NACE: ", provider_list[:25])
    # print("Example of provider NACE in 2 digit: ", provider_2digit_list[:25])
    print("Total provider NACE: {}, unique provider NACE: {}, unique 2-digit NACE: {}".format(
        len(provider_list), len(provider_unique), len(provider_2digit_unique)))
    print("Distribution of provider NACE sector: ", Counter(provider_sector_list))

    # get receiver NACE
    receiver_list = maestri_df["receiver_NACE"].tolist()
    receiver_list = [str(nace) for nace in receiver_list]
    receiver_unique = list(set(receiver_list))
    receiver_2digit_list = [nace[:2] for nace in receiver_list]
    receiver_2digit_unique = list(set(receiver_2digit_list))
    receiver_sector_list = [get_sector_from_code(nace_sector_dict, nace) for nace in receiver_list]
    # print("Example of provider NACE: ", receiver_list[:25])
    # print("Example of provider NACE in 2 digit: ", receiver_2digit_list[:25])
    print("Total receiver NACE: {}, unique receiver NACE: {}, unique 2-digit NACE: {}".format(
        len(receiver_list), len(receiver_unique), len(receiver_2digit_unique)))
    print("Distribution of receiver NACE sector: ", Counter(receiver_sector_list))

    # creat confusion matrix
    confusion_matrix = np.zeros((7, 7), dtype=int)
    alphabet_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "other": 6}
    provider_sector_list = [sector if sector in alphabet_to_index else "other" for sector in provider_sector_list]
    receiver_sector_list = [sector if sector in alphabet_to_index else "other" for sector in receiver_sector_list]
    for provider_sector, receiver_sector in zip(provider_sector_list, receiver_sector_list):
        confusion_matrix[alphabet_to_index[provider_sector], alphabet_to_index[receiver_sector]] += 1
    print("Confusion matrix for Maestri using its own sectors: \n", confusion_matrix)
    save_confusion_matrix(confusion_matrix, save_folder, filename_wo_extension="confusion_matrix_maestri", color_map="Oranges")

    if require_return:
        return waste_list, provider_sector_list, resource_list, receiver_sector_list


def parse_llm_result(result_str):
    digit_to_alphabet = {
        1: 'A',
        2: 'B',
        3: 'C',
        4: 'D',
        5: 'E',
        6: 'F'
    }
    match = re.search(r'(\d)', result_str)  # Matches the first digit anywhere in the string
    if match:
        digit = int(match.group(1))  # Gets the matched digit
        sector = digit_to_alphabet[digit] if digit in digit_to_alphabet else None
        return sector
    else:
        return None


def predict_sector_for_waste(model, waste):
    system_prompt = '''
    Given a material, you need to determine which industrial sector below may generate such material as output. Specify the index of the sector and provide an explanation within 30 words.
    If all sectors do not generate such material, answer with "None".
    Sector choices:
    1. Agriculture, forestry and fishing
    2. Mining and quarrying
    3. Manufacturing
    4. Electricity, gas, steam and air conditioning supply
    5. Water supply; sewerage, waste management and remediation activities
    6. Construction
    
    Here are some examples:
    Material: fruit peels.
    Answer: Sector 1. Fruit peels may be generated by fruit growing in agriculture industry.
    Material: heat.
    Answer: Sector 4. Heat may be generated by energy production.
    '''
    user_prompt = '''
    Material: {}. 
    Answer: '''.format(waste)

    llm_result = prompt_llm(model, system_prompt, user_prompt)
    llm_result_parsed = parse_llm_result(llm_result)   # either A/B/C... or None
    logging.info("Waste: {}. LLM output: '{}'. Parsed: {}".format(waste, llm_result, llm_result_parsed))

    return llm_result_parsed


def predict_sector_for_resource(model, resource):
    system_prompt = '''
    Given a material, you need to determine which industrial sector below may consume such material as input. Specify the index of the sector and provide an explanation within 30 words.
    If all sectors do not consume such material, answer with "None". 
    Sector choices:
    1. Agriculture, forestry and fishing
    2. Mining and quarrying
    3. Manufacturing
    4. Electricity, gas, steam and air conditioning supply
    5. Water supply; sewerage, waste management and remediation activities
    6. Construction
    
    Here are some examples:
    Material: hydrogen.
    Answer: Sector 4. Hydrogen may be used for electricity production.
    Material: raw material for fertilizer company.
    Answer: Sector 3. The raw material may be used for fertilizer manufacturing.
    '''
    user_prompt = '''
    Material: {}. 
    Answer: '''.format(resource)

    llm_result = prompt_llm(model, system_prompt, user_prompt)
    llm_result_parsed = parse_llm_result(llm_result)  # either A/B/C... or None
    logging.info("Resource: {}. LLM output: '{}'. Parsed: {}".format(resource, llm_result, llm_result_parsed))

    return llm_result_parsed


def assess_maestri_llm(model, save_folder):
    # set logging file
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    log_file = os.path.join(save_folder, "assess_maestri_{}.log".format(model))
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(message)s')
    logging.info("Start evaluating Maestri using {}...".format(model))

    # get data
    valid_sector = ["A", "B", "C", "D", "E", "F"]
    waste_list, provider_sector_list, resource_list, receiver_sector_list = assess_maestri(require_return=True)
    print(len(waste_list), len(provider_sector_list), len(resource_list), len(receiver_sector_list))   # all 425

    # remove duplicated
    unique_w2r = []
    unique_w2r_with_sector = []
    for waste, provider_sector, resource, receiver_sector in zip(waste_list, provider_sector_list, resource_list, receiver_sector_list):
        if (waste, resource) not in unique_w2r:
            unique_w2r.append((waste, resource))
            unique_w2r_with_sector.append((waste, provider_sector, resource, receiver_sector))
    waste_list = [item[0] for item in unique_w2r_with_sector]
    provider_sector_list = [item[1] for item in unique_w2r_with_sector]
    resource_list = [item[2] for item in unique_w2r_with_sector]
    receiver_sector_list = [item[3] for item in unique_w2r_with_sector]

    print("After removing duplicates:")
    print(len(waste_list), len(provider_sector_list), len(resource_list), len(receiver_sector_list))   # all 314
    print("Example of provider sector: ", provider_sector_list[:10])
    print("Example of resource: ", resource_list[:10])
    print("Example of receiver sector: ", receiver_sector_list[:10])

    # waste
    correct_predict_waste = 0
    incorrect_predict_waste = 0
    waste_sector_dict = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": []}
    other_sector = []
    for waste, provider_sector in tqdm.tqdm(zip(waste_list, provider_sector_list), total=len(waste_list), desc="Checking waste sectors"):
        predicted_sector = predict_sector_for_waste(model, waste)
        logging.info("Ground-truth sectors: {}, predicted sectors: {}".format(provider_sector, predicted_sector))
        if predicted_sector == provider_sector:
            correct_predict_waste += 1
        else:
            incorrect_predict_waste += 1
        # add to the dict
        if predicted_sector in waste_sector_dict.keys():
            waste_sector_dict[predicted_sector].append(waste)
        else:
            other_sector.append(waste)
    waste_sector_dict["other"] = other_sector
    waste_sector_count = {sector: len(waste_sector_dict[sector]) for sector in waste_sector_dict.keys()}
    print("Waste sector count: ", waste_sector_count)
    total_predict_waste = correct_predict_waste + incorrect_predict_waste
    print("Waste sector prediction correct: {}, incorrect: {}, total: {}".format(correct_predict_waste, incorrect_predict_waste, total_predict_waste))
    print("Waste sector accuracy: {}".format(correct_predict_waste / total_predict_waste))
    
    # resource
    correct_predict_resource = 0
    incorrect_predict_resource = 0
    resource_sector_dict = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": []}
    other_sector = []
    for resource, receiver_sector in tqdm.tqdm(zip(resource_list, receiver_sector_list), total=len(resource_list), desc="Checking resource sectors"):
        predicted_sector = predict_sector_for_resource(model, resource)
        logging.info("Ground-truth sectors: {}, predicted sectors: {}".format(receiver_sector, predicted_sector))
        if predicted_sector == receiver_sector:
            correct_predict_resource += 1
        else:
            incorrect_predict_resource += 1
        # add to the dict
        if predicted_sector in resource_sector_dict.keys():
            resource_sector_dict[predicted_sector].append(resource)
        else:
            other_sector.append(resource)
    resource_sector_dict["other"] = other_sector
    resource_sector_count = {sector: len(resource_sector_dict[sector]) for sector in resource_sector_dict.keys()}
    print("Resource sector count: ", resource_sector_count)
    total_predict_resource = correct_predict_resource + incorrect_predict_resource
    print("Resource sector prediction correct: {}, incorrect: {}, total: {}".format(correct_predict_resource, incorrect_predict_resource, total_predict_resource))
    print("Resource sector accuracy: {}".format(correct_predict_resource / total_predict_resource))

    # logging
    logging.info("Waste sector prediction correct: {}, incorrect: {}, total: {}".format(correct_predict_waste, incorrect_predict_waste, total_predict_waste))
    logging.info("Waste sector accuracy: {}".format(correct_predict_waste / total_predict_waste))

    logging.info("Resource sector prediction correct: {}, incorrect: {}, total: {}".format(correct_predict_resource, incorrect_predict_resource, total_predict_resource))
    logging.info("Resource sector accuracy: {}".format(correct_predict_resource / total_predict_resource))

    # construct the confusion matrix
    confusion_matrix = np.zeros((7, 7), dtype=int)
    alphabet_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "other": 6}
    for (waste, resource) in tqdm.tqdm(unique_w2r, desc="Constructing confusion matrix"):
        waste_sector = get_sector_from_dict(waste_sector_dict, waste)
        resource_sector = get_sector_from_dict(resource_sector_dict, resource)
        confusion_matrix[alphabet_to_index[waste_sector], alphabet_to_index[resource_sector]] += 1
    print("Confusion matrix: \n", confusion_matrix)

    # Save confusion matrix to file
    save_confusion_matrix(confusion_matrix, save_folder, filename_wo_extension="confusion_matrix_maestri_{}".format(model))


def assess_w2rkg_llm(model, save_folder):
    # use LLM to choose A-F, or "other" type
    print("Evaluating W2RKG coverage with {}...".format(model))

    # set logging file
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    log_file = os.path.join(save_folder, "w2rkg_logging_{}.log".format(model))
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(message)s')

    # read file and check statistics
    waste_list = []
    resource_list = []
    with open("result_all/after_fusion_v2/thre08_complete/fused_triples_aggregated.json") as file:
        data = json.load(file)
    for w2r in data:
        waste_list.append(w2r["waste"])
        resource_list.append(w2r["transformed_resource"])
    waste_unique_list = list(set(waste_list))
    resource_unique_list = list(set(resource_list))
    print("Total records: ", len(waste_list))
    print("Total unique waste: ", len(waste_unique_list))
    print("Total unique resource: ", len(resource_unique_list))

    ## check waste sector
    logging.info("Check for waste")
    print("Checking for waste...")
    waste_sector_dict = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": []}
    other_sector = []
    for waste in tqdm.tqdm(waste_unique_list):
        sector = predict_sector_for_waste(model, waste)   # either sector or None
        if sector is not None and sector in waste_sector_dict.keys():
            waste_sector_dict[sector].append(waste)
        else:
            other_sector.append(waste)
    waste_sector_dict["other"] = other_sector
    waste_sector_count = {sector: len(waste_sector_dict[sector]) for sector in waste_sector_dict.keys()}

    ## check resource sector
    logging.info("Check for resource")
    print("Checking for resource...")
    resource_sector_dict = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": []}
    other_sector = []
    for resource in tqdm.tqdm(resource_unique_list):
        sector = predict_sector_for_resource(model, resource)  # either sector or None
        if sector is not None and sector in resource_sector_dict.keys():
            resource_sector_dict[sector].append(resource)
        else:
            other_sector.append(resource)
    resource_sector_dict["other"] = other_sector
    resource_sector_count = {sector: len(resource_sector_dict[sector]) for sector in resource_sector_dict.keys()}

    print("Waste NACE sector count: ", waste_sector_count)
    print("Resource NACE sector count: ", resource_sector_count)

    # construct the confusion matrix
    confusion_matrix = np.zeros((7, 7), dtype=int)
    alphabet_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "other": 6}
    for w2r in data:
        waste = w2r["waste"]
        resource = w2r["transformed_resource"]
        waste_sector = get_sector_from_dict(waste_sector_dict, waste)
        resource_sector = get_sector_from_dict(resource_sector_dict, resource)
        confusion_matrix[alphabet_to_index[waste_sector], alphabet_to_index[resource_sector]] += 1
    print("Confusion matrix: \n", confusion_matrix)

    # Save confusion matrix to file
    save_confusion_matrix(confusion_matrix, save_folder, filename_wo_extension="confusion_matrix_w2rkg_{}".format(model))
    

def plot_num_of_entities(save_folder):
    # Maestri
    waste_list, provider_sector_list, resource_list, receiver_sector_list = assess_maestri(require_return=True)
    maestri_waste_num = len(list(set(waste_list)))
    maestri_resource_num = len(list(set(resource_list)))

    # W2RKG
    with open("result_all/after_fusion_v2/thre08_complete/fused_triples_aggregated.json") as file:
        data = json.load(file)
    waste_list = []
    resource_list = []
    for w2r in data:
        waste_list.append(w2r["waste"])
        resource_list.append(w2r["transformed_resource"])
    w2rkg_waste_num = len(list(set(waste_list)))
    w2rkg_resource_num = len(list(set(resource_list)))

    # plot
    plt.figure(figsize=(5, 8))
    x = np.arange(2)  # for 2 groups
    width = 0.35  # width of bars
    
    # Set font size
    plt.rcParams.update({'font.size': 16})
    
    # Create bars
    maestri_bars = plt.bar(x - width/2, [maestri_waste_num, maestri_resource_num], width, label='Maestri', color='tab:orange')
    w2rkg_bars = plt.bar(x + width/2, [w2rkg_waste_num, w2rkg_resource_num], width, label='W2RKG', color='tab:blue')
    
    # Add numbers on bars
    for bars in [maestri_bars, w2rkg_bars]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=16)
    
    plt.xticks(x, ['Waste', 'Resource'])
    plt.xlabel("      ")
    plt.ylabel("Number of unique entities")
    # plt.title("Number of unique entities in Maestri and W2RKG")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "num_of_entities.png"), dpi=300, bbox_inches='tight')


def plot_cm_from_csv(save_folder):
    # read csv
    maestri_llama_cm_df = pd.read_csv(os.path.join(save_folder, "confusion_matrix_maestri_llama3.1.csv"), index_col=0)
    maestri_gpt_cm_df = pd.read_csv(os.path.join(save_folder, "confusion_matrix_maestri_gpt-4o-mini.csv"), index_col=0)
    w2rkg_llama_cm_df = pd.read_csv(os.path.join(save_folder, "confusion_matrix_w2rkg_llama3.1.csv"), index_col=0)
    print(maestri_llama_cm_df)
    print(w2rkg_llama_cm_df)
    
    # Convert dataframes to numpy arrays
    maestri_llama_cm = maestri_llama_cm_df.to_numpy()
    maestri_gpt_cm = maestri_gpt_cm_df.to_numpy()
    w2rkg_llama_cm = w2rkg_llama_cm_df.to_numpy()

    # plot
    # if color_map = "coolwarm" for both, figures do not look nice
    save_confusion_matrix(maestri_llama_cm, save_folder, "confusion_matrix_maestri_llama3.1", color_map="Oranges")
    save_confusion_matrix(maestri_gpt_cm, save_folder, "confusion_matrix_maestri_gpt-4o-mini", color_map="Oranges")
    save_confusion_matrix(w2rkg_llama_cm, save_folder, "confusion_matrix_w2rkg_llama3.1", color_map="Blues")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run evaluation coverage analysis')
    parser.add_argument('--assess_maestri', action='store_true', help='Run Maestri evaluation')
    parser.add_argument('--assess_maestri_llm', action='store_true', help='Run Maestri LLM evaluation')
    parser.add_argument('--assess_w2rkg_llm', action='store_true', help='Run W2RKG LLM evaluation')
    parser.add_argument('--plot_num_of_entities', action='store_true', help='Plot number of entities')
    parser.add_argument('--plot_cm_from_csv', action='store_true', help='Plot confusion matrices from CSV')
    parser.add_argument('--model', default='llama3.1', choices=['llama3.1', 'gpt-4o-mini'], help='Model to use (default: llama3.1)')
    parser.add_argument('--folder', default='compare_Maestri', help='Output folder (default: compare_Maestri)')
    args = parser.parse_args()
    
    if args.assess_maestri:
        assess_maestri()

    if args.assess_maestri_llm:
        assess_maestri_llm(model=args.model, save_folder=args.folder)

    if args.assess_w2rkg_llm:
        assess_w2rkg_llm(model=args.model, save_folder=args.folder)

    if args.plot_num_of_entities:
        plot_num_of_entities(save_folder=args.folder)
        
    if args.plot_cm_from_csv:
        plot_cm_from_csv(save_folder=args.folder)
