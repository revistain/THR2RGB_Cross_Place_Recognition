import torch
from Parser import Parser
import logging
import os
from datetime import datetime
import torchvision.models as models
import numpy as np
from pathlib import Path

import commons
import utils
import inference
import datasets_T2R
import network
import backbone.dinov2.block as dinoblock

'''Setup'''
parser = Parser()
args = parser.parse_arguments()

# Set features_dim based on model
model_path = Path(args.foundation_model_path)
model_name = model_path.parts[-1].lower()
args.features_dim = 768 if 'vitb' in model_name else 384
dinoblock.adapter_dim = args.features_dim

commons.setup_logging(args.save_dir)
commons.seed_everything(args.seed)

utils.save_to_yaml(args)
logging.debug(f"The outputs are being saved in {args.save_dir}")

args.recall_values = list(range(1, 26))

'''Model'''
model = network.CrossModalVPR_Net(
    args,
    pretrained_foundation=True,
    foundation_model_path=args.foundation_model_path
)
model = model.to(args.device)

resume_path = args.resume[0]
print(f"Resuming from: {resume_path}")
model = utils.resume_model(resume_path, model)

# Wrap with DataParallel if multiple GPUs available
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

'''Dataset'''
DATASET_FOLDER = "./Dataset/save_mat"
for seq in args.test_seq if args.test_seq else args.sequences:
    logging.info(f"===== Evaluating Sequence: {seq} =====")
    args.sequences = [seq]
    test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')

    if args.use_reranking == 'gmrw':
        logging.info(f"Using GMRW re-ranking (top-{args.rerank_top_k}, score: {args.score_method})")
        recalls, recalls_str, recalls_before, recalls_str_before = \
            inference.inference_with_reranking(args, test_ds, model)
        logging.info(f"Recalls on {seq} (before re-ranking): {recalls_str_before}")
        logging.info(f"Recalls on {seq} (after re-ranking):  {recalls_str}")
    else:
        recalls, recalls_str = inference.inference(args, test_ds, model)
        logging.info(f"Recalls on {seq}: {recalls_str}")

    logging.info(f"================================================")
