import torch
import random
import utils
import network
import logging
import inference
import numpy as np
import datasets_T2R
from Parser import Parser

DATASET_FOLDER = "./Dataset/save_mat"

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def load_model(args, model):
    print(args.resume)
    args.resume = args.resume[0]
    model, _, _, start_epoch_num, _ = utils.resume_train(args, model, strict=False)
    print(f"Resuming from epoch {start_epoch_num}")

    args.sequences = ['SNU', 'Valley']
    test_sequences = args.sequences
    test_ds_list = []
    for seq in test_sequences:
        args.sequences = [seq]
        test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
        test_ds_list.append(test_ds)
    
    # Compute recalls
    model.eval()
    current_epoch_r1_list = []
    for seq, test_ds in zip(test_sequences, test_ds_list):
        logging.info(f"===== Evaluating Sequence: {seq} =====")
        args.current_epoch = start_epoch_num
        recalls, recalls_str = inference.inference(args, test_ds, model)
        logging.info(f"Recalls for {seq}: {recalls_str}")
        logging.info(f"================================================")
        current_epoch_r1_list.append(recalls[0])
        
if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_arguments()
    model = network.CrossModalVPR_Net(
        args,
        pretrained_foundation = True,
        foundation_model_path = args.foundation_model_path,
    )
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)
    
    load_model(args, model)