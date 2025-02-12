'''
This script is to augment the extraction_finetune_dataset.json
If wastes or resources have multiple items, shuffle them and add them to the dataset.
'''

import json
import random
from copy import deepcopy

# Set seed for reproducibility
random.seed(42)

# 1. Read the original dataset
with open('extraction_finetune_dataset.json', 'r') as f:
    original_dataset = json.load(f)

# Create a list to store all data (original + augmented)
augmented_dataset = []

# Add original data first
augmented_dataset.extend(original_dataset)


# Function to create shuffled versions of a list
def create_shuffled_versions(original_list):
    versions = []
    if len(original_list) >= 2:
        # Add first shuffled version
        first_shuffle = original_list.copy()
        random.shuffle(first_shuffle)
        versions.append(first_shuffle)
        
        # Add second shuffled version if more than 5 items
        if len(original_list) >= 4:
            second_shuffle = original_list.copy()
            random.shuffle(second_shuffle)
            versions.append(second_shuffle)
    return versions

# Create augmented versions
for entry in original_dataset:
    # Get shuffled versions of waste and resource lists
    waste_versions = create_shuffled_versions(entry['waste'])
    resource_versions = create_shuffled_versions(entry['transformed_resource'])
    
    # Create new entries for all combinations of shuffled versions
    for waste_list in waste_versions:
        for resource_list in resource_versions:
            new_entry = deepcopy(entry)
            new_entry['waste'] = waste_list
            new_entry['transformed_resource'] = resource_list
            augmented_dataset.append(new_entry)
        
        # If no resource versions, still add entries with just shuffled waste
        if not resource_versions:
            new_entry = deepcopy(entry)
            new_entry['waste'] = waste_list
            augmented_dataset.append(new_entry)
    
    # If no waste versions but have resource versions, add entries with just shuffled resources
    if not waste_versions:
        for resource_list in resource_versions:
            new_entry = deepcopy(entry)
            new_entry['transformed_resource'] = resource_list
            augmented_dataset.append(new_entry)

# 5. Save the augmented dataset
output_file = 'extraction_finetune_dataset_augmented.json'
with open(output_file, 'w') as f:
    json.dump(augmented_dataset, f, indent=4)

print(f"Original dataset size: {len(original_dataset)}")
print(f"Augmented dataset size: {len(augmented_dataset)}")
print(f"Saved augmented dataset to {output_file}")


