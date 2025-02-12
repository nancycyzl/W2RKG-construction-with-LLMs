import time
import tqdm
import json
import pandas as pd
import re
import os
from utils import *


def remove_head(text):
    introductions = [m.start() for m in re.finditer('Introduction', text, re.IGNORECASE)]

    if len(introductions) == 1:
        # If there is one occurrence, remove everything before it
        return text[introductions[0]:]
    elif len(introductions) > 1:
        # If there are two or more occurrences, remove everything before the second occurrence
        return text[introductions[1]:]
    else:
        # If "Introduction" is not found, return the original text
        return text


def remove_tail(text):
    references = [m.start() for m in re.finditer(r'\breference(?:s)?\b', text, re.IGNORECASE)]

    if len(references) == 1:
        # If there is one occurrence, remove everything after it
        return text[:references[0]]
    elif len(references) > 1:
        # If there are two or more occurrences, remove everything after the second occurrence
        return text[:references[1]]
    else:
        # If "reference" or "references" is not found, return the original text
        return text


def remove_bracket(text):
    # Regular expression to match content inside () or []
    # It handles nested brackets as well
    pattern = r'\([^()]*\)|\[[^\[\]]*\]'
    while re.search(pattern, text):  # Repeat until all nested brackets are removed
        text = re.sub(pattern, '', text)
    return text


def get_body(text):
    # step 1: remove beginning part: find the "Introduction" after the "corresponding author"
    text = remove_head(text)

    # step 2: remove ending part: find the "CRediT authorship" or "references" after the "conclusion"
    text = remove_tail(text)

    # step 3: remove text within {}, () or [] that is not just whitespace
    text = remove_bracket(text)

    return text


def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        fulltext = read_txt(file_path)
        body = get_body(fulltext)

        output_file = os.path.join(output_folder, filename)
        with open(output_file, 'w', encoding="utf-8") as file:
            file.write(body)


if __name__ == '__main__':
    main(input_folder="full_text_papers",
         output_folder="full_text_papers_processed")
