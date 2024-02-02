import argparse
import time
from utils import *
import pandas
import os
import warnings
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

mapper = {'citeseer': CiteseerGraphDataset, 'cora': CoraGraphDataset, 'pubmed': PubmedGraphDataset}
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
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--whereto', type=str, default=None)
parser.add_argument('--inductive', type=int, default=0)
args = parser.parse_args()

columns = ['name']
new_row = {}
datasets = ['citeseer', 'cora', 'pubmed']

if args.datasets is not None:
  datasets = args.datasets.split('-')
  print('Evaluated Datasets: ', datasets)

if args.models is not None:
    models = args.models.split('-')
    print('Evaluated Baselines: ', models)

for dataset in datasets:
    for metric in ['ACC mean', 'ACC std', 'Time']:
        columns.append(dataset+'-'+metric)

results = pandas.DataFrame(columns=columns)
file_id = None
for model in models:
    model_result = {'name': model}
    for dataset_name in datasets:
        transform = (AddSelfLoop())  # by default, it will first remove self-loops to prevent duplication
        time_cost = 0
        train_config = {
            'device': 'cuda',
            'epochs': 200,
            'patience': 50,
            'metric': 'ACC',
            'inductive': bool(args.inductive)
        }
        data = mapper[dataset_name](transform = transform)
        model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}

        acc_list = []
        for t in range(args.trials):
            torch.cuda.empty_cache()
            print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
            seed = seed_list[t]
            set_seed(seed)
            train_config['seed'] = seed
            detector = BaseNodeClassifierDetector(train_config, model_config, data)
            st = time.time()
            print(detector.model)
            test_score = detector.train()
            acc_list.append(test_score['ACC'])
            ed = time.time()
            time_cost += ed - st
        del detector, data

        model_result[dataset_name+'-ACC mean'] = np.mean(acc_list)
        model_result[dataset_name+'-ACC std'] = np.std(acc_list)
        model_result[dataset_name+'-Time'] = time_cost/args.trials
    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    file_id = save_results(results, file_id, args.whereto)
    print(results)
