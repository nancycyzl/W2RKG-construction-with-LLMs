'''
This script is a helper tool that facilitates the matching between Maestri and constructed W2RKG.
For each maestri record, input the waste / resource and specify the matching mode (1 for string match, 2 for semantic match)
The script will display the matched recorded from the W2RKG.
'''

import pandas as pd
from sentence_transformers import SentenceTransformer
from utils import *
from sklearn.metrics.pairwise import cosine_similarity


def create_Maestri_W2R():
    file_path = 'data/Maestri.xlsx'

    # Read the Excel file with header=[0, 1] to handle multi-level headers
    maestsri_df = pd.read_excel(file_path, header=[0, 1, 2])

    # Extract two columns: waste description and final use (transformed resource) and rename them
    maestsri_df.columns = ['_'.join(col).strip() for col in maestsri_df.columns.values]
    w2r_df = maestsri_df[['INVOLVED COMPANIES_Donor_Main business', 'INVOLVED COMPANIES_Receiver_Main business', 'EXCHANGE DESCRIPTION_Exchange Input_Waste description', 'EXCHANGE DESCRIPTION_Exchange details_Final use of the waste by the receiver company']].dropna()
    w2r_df.columns = ['donor_industry', 'receiver_industry', 'waste', 'transformed_resource']

    # if final use is "raw material", then change to "raw material for receiving_industry"
    w2r_df.loc[w2r_df['transformed_resource'] == 'Raw material', 'transformed_resource'] = 'Raw material for ' + w2r_df['receiver_industry']

    # save to excel
    w2r_df.to_excel('data/Maestri_W2R.xlsx', index=False)

    # statistics
    print("Total unique waste: ", w2r_df['waste'].nunique())   # 212
    print("Total unique transformed resource: ", w2r_df['transformed_resource'].nunique())  # 161

    w2r = [(waste, transformed_resource) for waste, transformed_resource in zip(w2r_df['waste'], w2r_df['transformed_resource'])]
    print("Total waste to resource pairs: ", len(w2r))                # 425
    print("Total unique waste to resource pairs: ", len(set(w2r)))    # 314

    # Create unique w2r pairs to a DataFrame
    w2r_df_new = pd.DataFrame(w2r, columns=['waste', 'transformed_resource'])
    unique_w2r_df = w2r_df_new.drop_duplicates()
    unique_w2r_df.to_excel('data/Maestri_W2R_unique.xlsx', index=False)
    print("length of unique w2r pairs: ", len(unique_w2r_df))

    # print(w2r_df.groupby('waste').size().sort_values(ascending=False))
    # print(w2r_df.groupby('transformed_resource').size().sort_values(ascending=False))


def create_embeddings(model, text_list):
    embedding = model.encode(text_list, convert_to_tensor=True)
    embedding_np = embedding.cpu().numpy()

    return embedding_np


def find_possible_matches():
    # this is used to help with matching Maestri W2R pairs with W2RKG
    # for each record in Maestri, use this script to filter out the possible matches in W2RKG
    # then manually check the matches and add them to Maestri_W2RKG_matched.xlsx

    # prepare model
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    # read W2RKG json file
    w2rkg_list = read_json('result_all/before_fusion_v2/all_w2r_list.json')   # a list of dict

    # create embeddings for wastes and resources
    print("Creating embeddings for W2RKG wastes and resources...")
    wastes = read_txt_as_list('result_all/before_fusion_v2/all_waste_list.txt')
    resources = read_txt_as_list('result_all/before_fusion_v2/all_resource_list.txt')
    embedding_wastes = create_embeddings(model, wastes)
    embedding_resources = create_embeddings(model, resources)
    print("Embeddings created")

    # set threshold
    threshold = 0.7

    while True:
        # ask for matching mode
        mode = input("Enter the matching mode: 1 for string match, 2 for embedding match (q to quit): ")

        # check if terminate
        if mode.lower() == 'q':
            break

        # ask for waste and resource
        waste = input("Enter the waste: ")
        resource = input("Enter the resource: ")

        if mode == '1':
            matched_wastes = [w for w in wastes if waste.lower() in w.lower()]
            matched_resources = [r for r in resources if resource.lower() in r.lower()]
            
        elif mode == '2':

            # create embeddings for waste and resource for test
            embedding_waste_test = create_embeddings(model, [waste])
            embedding_resource_test = create_embeddings(model, [resource])

            # Calculate cosine similarity for wastes
            waste_similarities = cosine_similarity(embedding_waste_test, embedding_wastes)[0]
            waste_matches = [(wastes[i], score) for i, score in enumerate(waste_similarities) if score > threshold]
            waste_matches.sort(key=lambda x: x[1], reverse=True)

            # Calculate cosine similarity for resources
            resource_similarities = cosine_similarity(embedding_resource_test, embedding_resources)[0]
            resource_matches = [(resources[i], score) for i, score in enumerate(resource_similarities) if score > threshold]
            resource_matches.sort(key=lambda x: x[1], reverse=True)

            # Get the matched wastes and resources
            matched_wastes = [match[0].lower() for match in waste_matches]
            matched_resources = [match[0].lower() for match in resource_matches]

        else:
            print("Invalid mode")
            continue

        # Find matching entries in w2rkg_list
        matching_entries = []
        for entry in w2rkg_list:
            waste_match = any(w.lower() in matched_wastes for w in entry['waste'])
            resource_match = any(r.lower() in matched_resources for r in entry['transformed_resource'])
            
            if waste_match and resource_match:
                matching_entries.append(entry)

        # Print results
        print("\nPossible matched entries from W2RKG:")
        for entry in matching_entries:
            print(f"Waste: {', '.join(entry['waste'])}")
            print(f"Resource: {', '.join(entry['transformed_resource'])}")
            print(f"source: {entry['reference']}")
            print("---")

        # print("matched_wastes: ", matched_wastes)
        # print("matched_resources: ", matched_resources)


if __name__ == "__main__":
    # create_Maestri_W2R()
    find_possible_matches()
