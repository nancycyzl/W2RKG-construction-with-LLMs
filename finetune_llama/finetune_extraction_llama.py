from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import json
import os
from trl import SFTTrainer
from transformers import TrainingArguments, TextIteratorStreamer
from unsloth import is_bfloat16_supported   # check if hardware supports bfloat16 precision
import time
import pandas as pd
import logging
import re
import ast
import argparse

from utils import check_make_dir

max_seq_length = 2048
dtype = None
load_in_4bit = True

# load llama base model
orig_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "E:/Models/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# apply peft LoRA to the base model
model = FastLanguageModel.get_peft_model(
    model = orig_model,
    r = 8,   # higher r allow model to learn more parameters
    target_modules = ["q_proj", "k_proj", "v_proj",   # query, key, value projections in the attention layers
                      "o_proj",      # output projection of the attention
                      "gate_proj", "up_proj", "down_proj"],  # related to gating mechanisms within the model
    lora_alpha = 32,   # control how much the learned low-rank matrics affect the original weights, alpha/rank
    lora_dropout = 0.2,
    bias = "none",    # none / all / lora_only
    use_gradient_checkpointing = "unsloth",  # unsloth / True. unsloth reduce VRAM by 30%
    random_state = 40,
    use_rslora = False,    # rank stablised LoRA
    loftq_config = None,   # LoftQ: low-rank and fine-tuned quantizatioin
)

# define some functions to prepare data for finetuning
def make_instruction_from_abstract():
  instruction_template = """Please read the following text and extract the waste-to-resource information into a json structure with keys of waste and transformed_resource.
The values should be list of strings. You should only use names or phrases to describe wastes and resources.
Keep the value empty if there is no corresponding information. Do not include any explanation. Do not include nested json content.
Follow the format:
{
   "waste": [],
   "transformed_resource": []
}
"""
  return instruction_template

def formatted_prompt(abstract, result)-> str:
  instruction = make_instruction_from_abstract()
  input_x = abstract
  response = json.dumps(result, indent=4)

  llama31_prompt="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

  return llama31_prompt.format(instruction, input_x, response)

def prepare_dataset(annotation_file):
  with open(annotation_file, "r", encoding="utf-8") as f:
    data = json.load(f)

  # format
  formatted_data = []
  for entry in data:
    abstract = entry["abstract"]
    result = {"waste": entry["waste"], "transformed_resource": entry["transformed_resource"]}

    EOS_TOKEN = tokenizer.eos_token  # must add this, otherwise the generation will not stop
    text = formatted_prompt(abstract, result) + EOS_TOKEN  # MAYBE DON'T NEED EOS_TOKEN
    formatted_data.append({"text":text})

  dataset = Dataset.from_list(formatted_data)
  return dataset


def get_finetune_dataset(annotation_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data = prepare_dataset(annotation_file)
    return train_data



train_data = get_finetune_dataset("finetune/annotation_extraction.json")

# prepare trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    dataset_text_field = "text",     # formatted_data.append({"text":text})
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,  # num of processes for dataset preprocessing, which speeds up data preparation
    packing = False,  # if True, combine shorter sequences to fill each batch, can increase training speed
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 5, # set this for 1 full training run,
        # max_steps = 60,     # otherwise, set maximum number of training steps
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 40,
        output_dir = "output",
        report_to = "none",  # can export to services like Weights & Biases or TensorBoard
    )
)

# train the model
model_name = "llama_ft_1"
base_dir = "finetune/extraction_finetune_unsloth"   # save to Gdrive

start_time = time.time()

trainer_stats = trainer.train()

end_time = time.time()
print(f"Training time: {(end_time - start_time)/60} minutes")


model.save_pretrained(os.path.join(base_dir, model_name))
tokenizer.save_pretrained(os.path.join(base_dir, model_name))
print("Finetuned model saved at: ", os.path.join(base_dir, model_name))



# define some functions for inference

def make_formatted_prompt_inference(abstract):
  propmt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

  {instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

  {input_x}<|eot_id|><|start_header_id|>assistant"""

  instruction = """Please read the following text and extract the waste-to-resource information into a json structure with keys of waste and transformed_resource.
The values should be list of strings. You should only use names or phrases to describe wastes and resources.
Keep the value empty if there is no corresponding information. Do not include any explanation. Do not include nested json content.
Follow the format:
{{
   "waste": [],
   "transformed_resource": []
}}
"""
  input_x = abstract

  formatted_prompt = propmt_template.format(instruction=instruction, input_x=input_x)
  return formatted_prompt

def extract(abstract, args):
  inputs_string = make_formatted_prompt_inference(abstract)
  inputs = tokenizer([inputs_string], return_tensors="pt").to("cuda")

  # Initialize the TextIteratorStreamer
  text_iter_streamer = TextIteratorStreamer(tokenizer)

  # generate
  _ = model.generate(**inputs, streamer=text_iter_streamer, max_new_tokens=512)
  generated_text = "".join([token for token in text_iter_streamer])

  # Remove the original input part from the generated text, if needed
  original_input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
  generated_text = generated_text[len(original_input_text):].strip()

  return generated_text

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
      required_keys = ["waste", "transformed_resource"]
      for key in required_keys:
          if key not in json_data:
              json_data[key] = []
      json_data["transforming_process"] = []   # add the transforming_process (placeholder)
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

    # get llm output and process
    coded_abstract = abstract   # no need to wrap
    result_str = extract(coded_abstract, args)
    result_json = extract_postprocessing(result_str)

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


def run_inference(args):
    # Run the process_abstracts function
    for i in range(args.repeat):
        print("Running experiment {}/{}".format(i+1, args.repeat))
        logging.info("Running experiment {}/{}".format(i+1, args.repeat))


        # create subfolder in the save_dir: run_{i}
        save_dir = os.path.join(args.result_base_folder, "run_{}".format(i))
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

        process_abstracts(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=50, help='How many abstracts to process. Set -1 to process all.')
    parser.add_argument('--start_index', type=int, help='Start index in the original csv file.')
    parser.add_argument('--model', type=str, default='llama3.1',
                        choices=["llama3", "llama3.1", "gpt-3.5-turbo", "gpt-4o-mini"],
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

