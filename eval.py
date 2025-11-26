import torch
from Parser import Parser
import logging
import os
from datetime import datetime
import torchvision.models as models
import numpy as np

import commons
import utils
import inference
# import datasets_dual
import datasets_T2R
import network

'''Setup'''
parser = Parser()
args = parser.parse_arguments()

commons.setup_logging(args.save_dir)
commons.seed_everything(args.seed)

utils.save_to_yaml(args)
logging.debug(f"The outputs are being saved in {args.save_dir}")

args.recall_values = list(range(1, 26))

'''Model'''
model = network.CrossModalVPR_Net(pretrained_foundation = True, foundation_model_path = args.foundation_model_path)
model = model.to(args.device)

resume_path= args.resume[0]
print(resume_path)
model = utils.resume_model(resume_path, model)

'''Dataset'''
DATASET_FOLDER = "./Dataset/save_mat"
for seq in args.sequences:
    logging.info(f"===== Evaluating Sequence: {seq} =====")
    args.sequences = [seq]
    test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
    recalls, recalls_str = inference.inference(args, test_ds, model)
    logging.info(f"Recalls on {seq}: {recalls_str}")
    logging.info(f"================================================")

