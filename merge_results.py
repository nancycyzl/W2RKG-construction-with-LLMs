from utils import *
import random


def get_abstract_results():
    # count how many W2R for abstract

    # REPLACE WITH YOUR FILES
    w2r_results = [
    'results_all/abstract/gpt-4o-mini_zero_code1/run_0/w2r_results.json',    # part one, from index 0 to 4490
    'results_all/abstract/gpt-4o-mini_zero_code1_1/run_0/w2r_results.json'   # part two, from index 4491 to end
    ]

    waste_list = []
    resource_list = []
    w2r_list = []   # a list of dict
    w2r_triple_count = 0
    
    for file in w2r_results:
        abstract_results = read_json(file)    # a list of dicts

        for w2r in abstract_results:
            if len(w2r['waste']) > 0 and len(w2r['transformed_resource']) > 0:
                waste_list.extend(w2r['waste'])
                resource_list.extend(w2r['transformed_resource'])
                w2r_triple_count += len(w2r['waste']) * len(w2r['transformed_resource'])
                if isinstance(w2r["reference"], float):   # if reference is float, means NaN, then assign empty string
                    w2r["reference"] = ""
                w2r_list.append(w2r)

    print("\n\nStatics for abstract:")
    print("number of W2R sets: ", len(w2r_list))
    print("number of unique wastes: ", len(set(waste_list)))
    print("number of unique resources: ", len(set(resource_list)))
    print("number of triples: ", w2r_triple_count)
    return waste_list, resource_list, w2r_list, w2r_triple_count


def get_fulltext_results():
    # REPLACE WITH YOUR FILES
    w2r_results = [
    'results_all/fulltext/gpt-4o-mini_zero_code1_chunk_2/review_papers'     # this is a folder, reference is doi + chunk id
    ]

    files = []
    for w2r_result_path in w2r_results:
        if os.path.isdir(w2r_result_path):
            for file in os.listdir(w2r_result_path):
                files.append(os.path.join(w2r_result_path, file))
        else:
            files.append(w2r_result_path)

    waste_list = []
    resource_list = []
    w2r_list = []  # a list of dict
    w2r_triple_count = 0

    for file in files:
        results = read_json(file)
        for result in results:
            if len(result['waste']) > 0 and len(result['transformed_resource']) > 0:
                waste_list.extend(result['waste'])
                resource_list.extend(result['transformed_resource'])
                w2r_triple_count += len(result['waste']) * len(result['transformed_resource'])
                if isinstance(result["reference"], float):   # if reference is float, means NaN, then assign empty string
                    result["reference"] = ""
                w2r_list.append(result)

    print("\n\nStatics for fulltext:")
    print("number of W2R sets: ", len(w2r_list))
    print("number of unique wastes: ", len(set(waste_list)))
    print("number of unique resources: ", len(set(resource_list)))
    print("number of triples: ", w2r_triple_count)
    return waste_list, resource_list, w2r_list, w2r_triple_count


def combine_results(save_folder):
    waste_list_abstract, resource_list_abstract, w2r_list_abstract, w2r_triple_count_abstract = get_abstract_results()
    waste_list_fulltext, resource_list_fulltext, w2r_list_fulltext, w2r_triple_count_fulltext = get_fulltext_results()

    # combine result for abstract and fulltext
    all_waste_list = list(set(waste_list_abstract + waste_list_fulltext))
    all_resource_list = list(set(resource_list_abstract + resource_list_fulltext))
    all_w2r_list = w2r_list_abstract + w2r_list_fulltext

    # save results
    save_list_to_txt(all_waste_list, os.path.join(save_folder, 'all_waste_list.txt'))
    save_list_to_txt(all_resource_list, os.path.join(save_folder, 'all_resource_list.txt'))
    save_json(all_w2r_list, os.path.join(save_folder, 'all_w2r_list.json'))
    
    # save a test file
    w2r_list_test = random.sample(all_w2r_list, 50)
    save_json(w2r_list_test, os.path.join(save_folder, 'all_w2r_list_test.json'))

    print("\n\nStatics for all:")
    print("number of W2R sets: ", len(all_w2r_list))
    print("number of unique wastes: ", len(all_waste_list))
    print("number of unique resources: ", len(all_resource_list))   
    print("number of triples: ", w2r_triple_count_abstract + w2r_triple_count_fulltext)


if __name__ == "__main__":
    combine_results(save_folder="results_all/before_fusion_v2")