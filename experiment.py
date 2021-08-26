import itertools
import os
import random
from argparse import ArgumentParser
from operator import itemgetter


def groupBy(l, key):
    all_annot = sorted(l, key=itemgetter(key))
    return dict((k, list(g)) for k, g in itertools.groupby(all_annot, key=itemgetter(key)))


def create_experiment(config):
    exp_dir = os.path.join(config.basedir, 'experiments', config.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    real_annots = []
    real_lines = open(config.real_img_csv, 'r').readlines()
    for real_img in real_lines:
        path, x1, y1, x2, y2, category = real_img.split(',')
        real_annots.append({'path': path, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'category': category})
    real = groupBy(real_annots, 'path')
    real_img_list = list(real.keys())[:config.n_real_imgs]
    random.shuffle(real_img_list)

    fake_annots = []
    fake_lines = open(config.fake_img_csv, 'r').readlines()
    for fake_img in fake_lines:
        path, x1, y1, x2, y2, category = fake_img.split(',')
        fake_annots.append({'path': path, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'category': category})
    fake = groupBy(fake_annots, 'path')
    fake_img_list = list(fake.keys())[:config.n_fake_imgs]
    random.shuffle(fake_img_list)

    img_annots_list = real_annots + fake_annots
    imgs_annots_map = groupBy(img_annots_list, 'path')
    mixed = real_img_list + fake_img_list
    random.shuffle(mixed)

    val_imgs = mixed[:config.n_val_size]
    train_imgs = mixed[config.n_val_size:]

    csv_train_file = os.path.join(exp_dir, 'train.csv')
    with open(csv_train_file, 'w') as f:
        for img in train_imgs:
            for a in imgs_annots_map[img]:
                f.write(a['path'] + ',' + str(a.get('x1', '')) + ',' + str(a.get('y1', '')) + ',' + str(
                    a.get('x2', '')) + ',' + str(a.get('y2', '')) + ',' + str(a.get('category', '')))

    csv_validation_file = os.path.join(exp_dir, 'validation.csv')
    with open(csv_validation_file, 'w') as f:
        for img in val_imgs:
            for a in imgs_annots_map[img]:
                f.write(a['path'] + ',' + str(a.get('x1', '')) + ',' + str(a.get('y1', '')) + ',' + str(
                    a.get('x2', '')) + ',' + str(a.get('y2', '')) + ',' + str(a.get('category', '')))


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract annotations for RetinaNET")
    parser.add_argument('--real_img_csv', required=True)
    parser.add_argument('--fake_img_csv', required=True)
    parser.add_argument('--basedir', required=True)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--n_val_size', required=True, type=int)
    parser.add_argument('--n_real_imgs', required=True, type=int)
    parser.add_argument('--n_fake_imgs', required=True, type=int)
    create_experiment(parser.parse_args())
