import requests
import pandas as pd
import tqdm
import os

API_KEY = '00000'   # REPLACE WITH YOUR ELSEVIER API KEY
SAVE_DIR = 'full_text_papers'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# link should be format like: https://api.elsevier.com/content/article/doi/10.1016/j.ibusrev.2010.09.002

'''
By default, a request to https://api.elsevier.com/content/article/... will return:
- the full-text XML version (FULL view) of the document, if your client IP address is recognized as that of a
subscribing institute that has access to that article on ScienceDirect (which we call 'being entitled to the document');
- the abstract XML version (META_ABS) of the document, if you're not entitled.
'''


def download_full_text(doi):
    full_text_url = f'https://api.elsevier.com/content/article/doi/{doi}'
    # print(full_text_url)
    headers = {
        'X-ELS-APIKey': API_KEY,
        # 'Accept': 'application/json',
        "Accept": "text/plain",   # for stripped-down full-text
    }

    response = requests.get(full_text_url, headers=headers)
    if response.status_code == 200:
        paper_content = response.text
        # print(paper_content)

        save_path = os.path.join(SAVE_DIR, f'{doi.replace("/", "_")}.txt')
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(str(paper_content))  # Adjust based on the format you receive
            print("Success!!")
        return 1
        # print(f"Downloaded {doi}")
    else:
        print("Fail, ", "Status code:", response.status_code)
        return response.status_code


def main(file=None, num=None):
    # Test with one example
    # doi = "10.3389/frsus.2024.1300904"
    # download_full_text(doi)

    success = 0
    not_found = 0
    not_oa = 0
    other = 0

    # download all as in csv file
    df = pd.read_csv(file)
    doi_list = df["DOI"].tolist()  # some do not have DOI
    if num:
        doi_list = doi_list[:num]  # try only a small portion, the successfully downloaded papers will be much less

    for paper_id, doi in enumerate(doi_list):
        print("Paper {} / {}: {}".format(paper_id, len(doi_list), doi), end="  ")
        if doi:
            result = download_full_text(str(doi))

            # record result
            if result == 1:
                success += 1
            elif result == 404:
                not_found += 1
            elif result == 400:
                not_oa += 1
            else:
                other += 1

    print("Success: {},  not found: {},  not OA: {},  other: {}".format(success, not_found, not_oa, other))


if __name__ == "__main__":
    csv_file = "scopus_waste2resource.csv"
    main(file=csv_file, num=50)
