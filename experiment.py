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

def setup_test(config):
    exp_path = os.path.join(config.basedir, config.exp_name)
    if not os.path.isdir(exp_path):
        print('Experiment directory doesn\'t exist: ', exp_path)
        exit()
    test_file = os.path.join(exp_path, 'test.csv')
    if os.path.isfile(test_file):
        print('Test file already exists for this experiment')
        exit()
    with open(test_file, 'w') as f:
        for img in os.listdir(config.test_img_path):
            f.write(os.path.join(config.test_img_path, img) + '\n')
        f.close()
    print('Created test file: ', test_file)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract annotations for RetinaNET")
    subparsers = parser.add_subparsers()

    create_subparser = subparsers.add_parser('create')
    create_subparser.add_argument('--real_img_csv', required=True)
    create_subparser.add_argument('--fake_img_csv', required=True)
    create_subparser.add_argument('--basedir', required=True)
    create_subparser.add_argument('--exp_name', required=True)
    create_subparser.add_argument('--n_val_size', required=True, type=int)
    create_subparser.add_argument('--n_real_imgs', required=True, type=int)
    create_subparser.add_argument('--n_fake_imgs', required=True, type=int)
    create_subparser.set_defaults(func=create_experiment)

    setup_test_subparser = subparsers.add_parser('setup-test')
    setup_test_subparser.add_argument('--exp_name', required=True)
    setup_test_subparser.add_argument('--basedir', default='experiments', help='The base experiments directory')
    setup_test_subparser.add_argument('--test_img_path', required=True)
    setup_test_subparser.set_defaults(func=setup_test)

    args = parser.parse_args()
    args.func(args)