from pathlib import Path
import json

def get_config():
    return {
        "batch_size": 100,
        "num_epochs": 15,
        "lr": 10**-4,
        "d_model": 512,
        "datasource": 'text2sql',  # Update with a keyword to indicate local data
        "model_folder": "weights",
        "model_basename": "model_",  # Update with your desired model file name prefix
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/model",
        "seq_len_src":82,
        "seq_len_tgt":82,
        "qst_train_vocab":dict(),
        "qer_train_vocab":dict()
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


def save_config(config, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(config, json_file)