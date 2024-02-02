import argparse
import time
import pandas as pd
import os
import warnings
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dgl import DGLGraph
from dgl.transform import add_self_loop
from dgl.data import DGLDataset
from ogb.nodeproppred import DglNodePropPredDataset
from models.detector import BaseNodeClassifierDetector

warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--whereto', type=str, default=None)
parser.add_argument('--inductive', type=int, default=0)

args = parser.parse_args()

columns = ['name']
new_row = {}
dataset_names = ['ogbn-arxiv', 'ogbn-products']

if args.models is not None:
    models = args.models.split('-')
    print('Evaluated Baselines: ', models)

for dataset_name in dataset_names:
    columns.append(dataset_name + '-Time')

results = pd.DataFrame(columns=columns)
file_id = None

for model in models:
    model_result = {'name': model}
    for dataset_name in dataset_names:
        time_cost = 0
        train_config = {
            'device': 'cuda',
            'epochs': 200,
            'patience': 10000,
            'metric': 'ACC',
            'inductive': bool(args.inductive)
        }
        dataset = DglNodePropPredDataset(name=dataset_name)
        graph, labels = dataset[0]
        graph = add_self_loop(graph)
        data = DataLoader([(graph, labels)], batch_size=1, shuffle=True)
        model_config = {'model': model, 'lr': 0.01, 'dropout_rate': 0.5, 'h_feats': 16}
        acc_list = []

        for t in range(args.trials):
            torch.cuda.empty_cache()
            print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
            seed = seed_list[t]
            set_seed(seed)
            train_config['seed'] = seed
            detector = BaseNodeClassifierDetector(train_config, model_config, graph, labels)
            st = time.time()
            print(detector.model)
            test_score = detector.train()
            acc_list.append(test_score['ACC'])
            ed = time.time()
            time_cost += ed - st

        del detector, data

        model_result[dataset_name + '-ACC mean'] = np.mean(acc_list)
        model_result[dataset_name + '-ACC std'] = np.std(acc_list)
        model_result[dataset_name + '-Time'] = time_cost / args.trials

    model_result = pd.DataFrame(model_result, index=[0])
    results = pd.concat([results, model_result])
    file_id = save_results(results, file_id, args.whereto)
    print(results)
