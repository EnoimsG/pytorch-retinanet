import argparse
import json
import os

import pandas as pd
import wandb

api = wandb.Api()
entity = 'simonegori'
exp_name = 'retinanet-synththerm-mixed'
run_path = entity + '/' + exp_name


# Ex: results_yolo_masked_real_80_fake_20
# Ex: results_retinanet50_masked_10
def parse_exp_name(exp_name):
    s = exp_name.split('_')
    return s[2], s[3]


def extract():
    # 1. Download tables into media/table as json files
    rs = api.runs(run_path)
    for run in rs:
        if not run.name.startswith('results'):
            continue
        print('Downloading results for run:', run.name)
        fname = (run.history()['results']).to_list()[0]['path']
        f = run.file(fname).download(root='experiments_results_csv', replace=True)  # put in media/table
        f.close()
        os.rename('experiments_results_csv/' + fname, 'experiments_results_csv/media/table/' + run.name + '.json')

    # 2. Parse JSON
    frames = []
    for json_file in os.listdir('experiments_results_csv/media/table'):
        exp_name = json_file.split('.')[0]
        j = json.load(open('experiments_results_csv/media/table/' + json_file, 'r'))
        gan_type, p_increment = parse_exp_name(exp_name)
        columns = ['exp_name', 'gan_type', 'p_increment'] + j['columns']
        for el in j['data']:
            el.insert(0, exp_name)
            el.insert(1, gan_type)
            el.insert(2, p_increment)
        df = pd.DataFrame(data=j['data'], columns=columns)
        frames.append(df)
    df = pd.concat(frames)
    df.to_csv('experiments_results_csv/retinanet-synththerm-mixed.csv', index=False)
    print('done')


if __name__ == '__main__':
    extract()
