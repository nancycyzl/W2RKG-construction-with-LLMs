import json
import random
from copy import deepcopy
import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


random.seed(55)

# get the w2r csv file
df = pd.read_csv("scopus_waste2resource.csv")


def obtain_text_from_abstract(doi):
    row = df[df['DOI'] == doi]
    if len(row) == 0:
        raise ValueError(f"DOI {doi} not found in W2R CSV file!")
    return row['Abstract'].values[0]


def obtain_text_from_chunk(reference, chunk_size=1000):
    doi, chunk_id = reference.split('_chunk')
    chunk_id = int(chunk_id)
    filename = os.path.join("full_text_papers_processed", doi.replace('/', '_') + '.txt')

    # make sure the file exists
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        chunks = split_into_chunks(content, chunk_size)
        return chunks[chunk_id]
    else:
        raise ValueError(f"File {filename} does not exist!")


def split_into_chunks(text, chunk_size):
    length_function = len

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "."],
        chunk_size=chunk_size,
        chunk_overlap=100,
        length_function=length_function,
    )
    splits = splitter.split_text(text)

    return splits


def main(base_folder, num_samples=100):
    with open(os.path.join(base_folder, 'fused_triples_aggregated.json'), 'r') as f:
        triples = json.load(f)

    # Create lists to store data for DataFrame
    data = []
    sampled_indices = set()  # Keep track of sampled indices
    used_references = set()  # Keep track of used references
    
    while len(data) < num_samples:
        # Sample one triple that hasn't been sampled before
        available_indices = list(set(range(len(triples))) - sampled_indices)

        idx = random.choice(available_indices)
        sampled_indices.add(idx)
        triple = triples[idx]
        
        # Skip if reference is empty
        if not triple['reference']:
            continue
            
        waste = triple['waste']
        resource = triple['transformed_resource']
        process = triple['transforming_process']
        references = triple['reference']  # a string of references separated by comma
        
        try:
            # get one reference that hasn't been used
            available_refs = [ref for ref in references.split(', ') if ref not in used_references]
            if not available_refs:
                continue
                
            reference = random.choice(available_refs)
            used_references.add(reference)
            print("Processing reference:", reference)

            # check if reference is abstract or chunk
            if "chunk" in reference:
                text = obtain_text_from_chunk(reference)   # reference should be DOI_chunk[id]
                text_type = "chunk"
            else:
                text = obtain_text_from_abstract(reference) # reference should be DOI
                text_type = "abstract"

            # Add data to list
            data.append({
                'text': text,
                'text_type': text_type,
                'reference': reference,
                'waste': waste,
                'resource': resource,
                'process': process,
                'waste_aligned': '',  # empty value
                'waste_valid': '',  # empty value
                'resource_aligned': '',  # empty value
                'resource_valid': '',  # empty value
                'process_aligned': '',  # empty value
                'remarks': ''  # empty value
            })
        except Exception as e:
            print(f"Error processing reference {reference}: {str(e)}")
            continue

    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    output_file = os.path.join(base_folder, 'kg_test_dataset.xlsx')
    df.to_excel(output_file, index=False)
    
    print(f"Created test dataset with {len(data)} entries")
    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main(base_folder='result_all/after_fusion_v2/thre08_complete', num_samples=100)

