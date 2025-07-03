import os
import sys
import h5py
import json
import numpy as np
import torch as pt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

from src.dataset import StructuresDataset, collate_batch_features, select_by_sid, select_by_interface_types
from src.data_encoding import encode_structure, encode_features, extract_topology, categ_to_resnames, resname_to_categ
from src.structure import data_to_structure, encode_bfactor, concatenate_chains, split_by_chain
from src.structure_io import save_pdb, read_pdb
from src.scoring import bc_scoring, bc_score_names


# model parameters
save_path = "/PeSTo/model/save/i_v4_1_2021-09-07_11-21"  # 91

# select saved model
model_filepath = os.path.join(save_path, 'model_ckpt.pt')

# add module to path
if save_path not in sys.path:
  sys.path.insert(0, save_path)

# import functions
from config import config_model, config_data
from data_handler import Dataset
from model import Model

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_inference.py <pdb input folder> <output folder>")
        exit(1)


    # define device
    if not pt.cuda.is_available():
        print("[WARNING] CUDA not available. Falling back to cpu")
        device = pt.device("cpu")
    else:
        device = pt.device("cuda")

    input_folder = sys.argv[1]
    output_root = sys.argv[2]

    os.makedirs(output_root, exist_ok=True)

    # create model
    model = Model(config_model)

    # reload model
    model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cuda")))

    # set model to inference
    model = model.eval().to(device)

    # find pdb files and ignore already predicted ions
    pdb_filepaths = [fp for fp in glob(os.path.join(input_folder, "*.pdb"), recursive=True) if "_i" not in fp]

    # create dataset loader with preprocessing
    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)

    # run model on all subunits
    with pt.no_grad():
        for subunits, filepath in tqdm(dataset):
            sample_name = '.'.join(os.path.basename(filepath).split('.')[:-1])

            # concatenate all chains together
            structure = concatenate_chains(subunits)

            # encode structure and features
            X, M = encode_structure(structure)
            q = encode_features(structure)[0]

            # extract topology
            ids_topk, _, _, _, _ = extract_topology(X, 64)

            # pack data and setup sink (IMPORTANT)
            X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

            # run model
            z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))

            p = pt.sigmoid(z[:,0])

            # encode result
            structure = encode_bfactor(structure, p.cpu().numpy())
            # save results
            output_filepath = os.path.join(output_root, f'{sample_name}_if.pdb')
            save_pdb(split_by_chain(structure), output_filepath)


if __name__ == "__main__":
    main()