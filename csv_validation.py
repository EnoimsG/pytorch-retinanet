import argparse
import os

import numpy as np
import torch
from torchvision import transforms

import wandb
from retinanet import csv_eval
from retinanet.dataloader import CSVDataset, Resizer, Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def validate_one(args, model_path):
    # dataset_val = CocoDataset(args.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(args.csv_annotations_path, args.class_list_path,
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    # retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet = torch.load(model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        # retinanet.load_state_dict(torch.load(args.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(args.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    aps, results = csv_eval.evaluate(dataset_val, retinanet, iou_threshold=float(args.iou_threshold), test=True)
    print("results: ", results)
    return results


def validate_and_send(config):
    results = []
    weights_path = os.path.join(config.exp_dir, 'weights')
    for w_file in os.listdir(weights_path):
        if 'final' in w_file:  # already included
            continue
        epoch_results = validate_one(config, os.path.join(weights_path, w_file))
        for er in epoch_results:
            results.append({'epoch': int(w_file.split('_')[2].split('.')[0]), 'class': er['label'],
                            'ap': er['ap'], 'noccur': er['na']})
    e = {}
    for r in results:
        if r['epoch'] not in e:
            e[r['epoch']] = []
        e[r['epoch']].append(r['ap'])
    maps = {}
    for k, v in e.items():
        maps[k] = np.mean(v)

    flatten_results = []
    for r in results:
        flatten_results.append([r['epoch'], r['class'], r['ap'], r['noccur'], maps[r['epoch']]])
    run_name = os.path.basename(os.path.normpath(config.exp_dir))
    run = wandb.init(project=config.wandb_name, config={'exp_name': 'results_' + run_name})
    wandb.run.name = 'results_' + run_name
    wandb.run.save()
    result_table = wandb.Table(columns=["epoch", "class", "ap", "noccur", "map"], data=flatten_results)
    wandb.log({'results': result_table})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--wandb_name', required=True, help='Wandb name')
    parser.add_argument('--exp_dir', help='The experiment name')
    parser.add_argument('--all', required=False, type=bool, default=False, help='Should compute all')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    # parser.add_argument('--images_path', help='Path to images directory', type=str)
    parser.add_argument('--class_list_path', help='Path to classlist csv', type=str)
    parser.add_argument('--iou_threshold', help='IOU threshold used for evaluation', type=str, default='0.5')
    args = parser.parse_args()
    validate_and_send(args)
