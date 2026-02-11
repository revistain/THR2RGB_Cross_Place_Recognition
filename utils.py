import os
import yaml
import torch
from collections import OrderedDict
from datetime import datetime
import shutil

def save_to_yaml(args, filename='config.yaml'):
    file_path = os.path.join(args.save_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.dump(vars(args), file, default_flow_style=False)
        
def save_checkpoint(args, state, is_best, filename):
    model_path = os.path.join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(args.save_dir, "best_model.pth"))

def resume_model(resume_path, model, optimizer=None, strict=False):
    checkpoint = torch.load(resume_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model

def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    if type(args.resume) is list and len(args.resume) != 0:
        args.resume = args.resume[0]
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
        
    start_epoch_num = checkpoint["epoch_num"]
    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    
    print(f"Weights loaded.")
    print(f"- Missing keys: {len(missing)}")
    print("    - Missing: ", missing)
    print(f"- Unexpected keys (should be 0): {len(unexpected)}")
    print("    - unexpected: ", unexpected)
    
    if optimizer:
        optimizer.load_state_dict(new_state_dict)
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    not_improved_num = checkpoint["not_improved_num"]
    return model, optimizer, None, start_epoch_num, not_improved_num

cached_timestamp = None
def get_timestamp():
    global cached_timestamp
    if cached_timestamp is None:
        cached_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return cached_timestamp