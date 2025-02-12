from nltk.stem import WordNetLemmatizer
import os
import json
import pandas as pd
import ollama
from openai import OpenAI


def prompt_llm(model, system_prompt, user_prompt):
    if "llama" in model.lower():
        response = ollama.chat(
            model=model,
            messages=[
              {
                "role": "system",
                "content": system_prompt,
              },
              {
                "role": "user",
                "content": user_prompt,
              }
            ])

        response_string = response['message']['content']
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

        response_string = response.choices[0].message.content
    else:
        raise ("Model {} is not supported!".format(model))

    return response_string


def check_make_dir(directory, exist_ok=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return directory
    else:
        if not exist_ok:
            i = 1
            # increment until finding an available name
            new_dir = f"{directory}_{i}"
            while os.path.exists(new_dir):
                i += 1
                new_dir = f"{directory}_{i}"
            os.makedirs(new_dir)
            return new_dir
        return directory


def read_txt_as_list(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
        return lines


def read_txt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        return content

def save_list_to_txt(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(f"{item}\n")

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def save_metrics_as_json(TP_list, FN_list, FP_list, jaccard_list, file_path):
    data = {
        "TP": TP_list,
        "FN": FN_list,
        "FP": FP_list,
        "jaccard": jaccard_list
    }

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def remove_plural_to_lower(words):
    lemmatizer = WordNetLemmatizer()

    def remove_plural_string(word):
        if word:
            word_split = word.split()  # eg. split "cocoa shells" into [cocoa, shells]
            word_split[-1] = lemmatizer.lemmatize(word_split[-1], pos='n')   # eg. shells -> shell
            return " ".join(word_split)
        else:
            return ""

    if isinstance(words, str):
        return remove_plural_string(words.lower())
    if isinstance(words, list):
        return [remove_plural_string(word.lower()) for word in words]


def save_metrics_df(metrics_list, mode='w', filepath='', comment=''):
    # mode: a - append, w - write

    metrics = metrics_list.copy()

    headers = ['micro_pre', 'micro_rec', 'micro_f1', 'macro_pre', 'macro_rec', 'macro_f1', 'avg_jaccard', 'comment']
    if len(metrics) != 7:
        raise Exception("The metrics list should have exactly 7 elements. Metrics has {} elements: {}".format(len(metrics), metrics))

    if mode == "a":
        if filepath is None:
            raise ValueError("filepath cannot be None")
        if comment is None:
            raise ValueError("comment cannot be None. Specify the model/prompting in the comment for comparation.")

        # if the file does not exist, create an empty csv file with the header
        if not os.path.exists(filepath):
            df = pd.DataFrame(columns=headers)
            df.to_csv(filepath, index=False)

        metrics.append(comment)
        df_existing = pd.read_csv(filepath)
        df_existing.loc[len(df_existing)] = metrics

        df_existing.to_csv(filepath, index=False)

    elif mode == "w":
        if filepath is None:
            filepath = "metrics_compare.csv"
        metrics.append(comment)  # add the comment column
        df = pd.DataFrame([metrics], columns=headers)
        df.to_csv(filepath, index=False)


def save_2d_list(data, headers, filepath):
    if len(data) != 0:
        assert len(data[0]) == len(headers)
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(filepath, index=False)
