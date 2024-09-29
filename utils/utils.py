import urllib.request
import os
import json
import torch
import yaml
import numpy as np
import random

# Function to download the file
def download_file(url, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    urllib.request.urlretrieve(url, filename)
    print(f"[{filename}] downloaded successfully!")

# Ask for user input (yes/no)
def ask_to_download(file_url, filename):
    if os.path.exists(filename):
        print(f"{filename} exists okay.")
        return
    
    response = input(f"Do you want to download the file? [{file_url.split('/')[-1]}] (yes/no): ")
    
    if response == 'yes':
        download_file(file_url, filename)
    elif response == 'no':
        print("File download canceled.")
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
        ask_to_download(file_url, filename)


def to_json(path_params, path_json):
    params = torch.load(path_params, map_location=torch.device('cpu'))
    raw_state_dict = {}
    for k, v in params.items():
        val = v.flatten().numpy().tolist()
        raw_state_dict[k] = val

    with open(path_json, 'w') as outfile:
        json.dump(raw_state_dict, outfile,indent= "\t")
        
class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
        
def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    return args
        
def get_network_paras_amount(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info[model_name] = trainable_params
    return info


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)