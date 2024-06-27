import json

def save_dict_to_json(dict_variable, file_path):
    with open(file_path, "w") as f:
        json.dump(dict_variable, f)


def calculate_f1_score(p, r):
    return (2 * p * r) / (p + r)

def explode_list(nested_list):
    return [item for sublist in nested_list for item in sublist]