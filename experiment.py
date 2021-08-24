"""
Generates annottaion for FLIR and improved dataset
"""
import itertools
import json
import os
import random
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from operator import itemgetter

import cv2

label_map = {
    'Person': 1,
    'Car': 3
}


def process_real_imgs(config):
    json_files = [f for f in os.listdir(config.real_labels_dir) if f.endswith('.json')]
    result = []
    for i, json_file in enumerate(json_files):
        if i >= config.n_real_imgs:
            break
        f = open(os.path.join(config.real_labels_dir, json_file), 'r')
        parsed_json = json.load(f)
        annotations = parsed_json['annotation']
        if len(annotations) == 0:
            result.append({'path': os.path.join(config.real_img_dir, parsed_json['image']['file_name'] + '.jpeg')})
        else:
            for a in annotations:
                x1 = int(a['bbox'][0])
                x2 = x1 + int(a['bbox'][2])
                y1 = int(a['bbox'][1])
                y2 = y1 + int(a['bbox'][3])
                result.append(
                    {'path': os.path.join(config.real_img_dir, parsed_json['image']['file_name'] + '.jpeg'), 'x1': x1,
                     'x2': x2, 'y1': y1, 'y2': y2, 'category': a['category_id']})
    return result


def process_fake_dataset(folder, name, config, processed):
    dataset_annotations = get_annotations_xml_from_folder(folder)
    img_dir = os.path.join(folder, name + '_insert')
    results = []
    for i, img_name in enumerate(os.listdir(img_dir)):
        if processed + i >= config.n_fake_imgs:
            break
        frame_number = '{:05d}'.format(int(img_name.split("_")[3].split(".")[0]))
        img = cv2.imread(os.path.join(img_dir, img_name))
        img_height, img_width, _ = img.shape
        # with open(os.path.join(config.outdir, img_name.split(".")[0] + ".txt"), "w") as f:
        for track in dataset_annotations:
            data_frames = list(filter(lambda x: '{:05d}'.format(int(x.attrib['frame'])) == frame_number,
                                      list(track)))  # get data element where frame = counter
            for d in data_frames:
                x1 = round(float(d.attrib['xtl']))
                x2 = round(float(d.attrib['xbr']))
                y1 = round(float(d.attrib['ytl']))
                y2 = round(float(d.attrib['ybr']))
                if x1 + x2 == 0 and x1 - x2 == 0:
                    results.append({'path': os.path.join(img_dir, img_name)})
                else:
                    results.append({'path': os.path.join(img_dir, img_name), 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
                                    'category': get_label(track)})
    return results, i


def get_label(track):
    """
    Return the appropriate label for the track, this exists because of some inconsistencies in the dataset
    :param label_attribute:
    :return:
    """
    label_attrib = track.attrib['label']
    if not label_attrib:
        label_name = track.attrib['name']
        if label_name == 'Suzuki_Swift':  # Inconsistency in FLIR00039 dataset
            return label_map['Car']
        else:
            raise Exception('Unrecognized label')
    return label_map[label_attrib]


def get_annotations_xml_from_folder(folder):
    xml_name = [fn for fn in os.listdir(folder) if fn.endswith(".xml")][0]  # get xml name
    tree = ET.parse(os.path.join(folder, xml_name))
    return tree.getroot()


def process_fake_imgs(config):
    elements = [e for e in os.listdir(config.fake_dir) if e.startswith("FLIR")]
    results = []
    processed = 0
    for e in elements:
        if e == 'FLIR00498':
            continue
        r, new_processed = process_fake_dataset(os.path.join(config.fake_dir, e), e, config, processed)
        processed += new_processed
        results.extend(r)
    return results


def create_mixed_exp(config):
    exp_path = os.path.join(config.basedir, config.exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    real_annot = process_real_imgs(config)
    fake_annot = process_fake_imgs(config)

    all_annot = real_annot + fake_annot
    all_annot = sorted(all_annot, key=itemgetter('path'))
    img_annots_map = dict((k, list(g)) for k, g in itertools.groupby(all_annot, key=itemgetter('path')))
    img_list = list(img_annots_map.keys())
    random.shuffle(img_list)
    train_size = int(len(img_list) - config.val_size)
    train_imgs = img_list[:train_size]
    val_imgs = img_list[train_size:]

    csv_train_file = os.path.join(exp_path, 'train.csv')
    with open(csv_train_file, 'w') as f:
        for img in train_imgs:
            for a in img_annots_map[img]:
                f.write(a['path'] + ',' + str(a.get('x1', '')) + ',' + str(a.get('y1', '')) + ',' + str(
                    a.get('x2', '')) + ',' + str(a.get('y2', '')) + ',' + str(a.get('category', '')) + '\n')

    csv_validation_file = os.path.join(exp_path, 'validation.csv')
    with open(csv_validation_file, 'w') as f:
        for img in val_imgs:
            for a in img_annots_map[img]:
                f.write(a['path'] + ',' + str(a.get('x1', '')) + ',' + str(a.get('y1', '')) + ',' + str(
                    a.get('x2', '')) + ',' + str(a.get('y2', '')) + ',' + str(a.get('category', '')) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract annotations for RetinaNET")
    subparsers = parser.add_subparsers()

    create_parser = subparsers.add_parser('create-mixed-exp', help='Creates a mixed experiment')
    create_parser.add_argument('--real_img_dir', type=str, required=True)
    create_parser.add_argument('--real_labels_dir', type=str, required=True)
    create_parser.add_argument('--fake_dir', type=str, required=True, help='Contains both real and fake')
    create_parser.add_argument('--n_real_imgs', type=int, required=True)
    create_parser.add_argument('--n_fake_imgs', type=int, required=True)
    create_parser.add_argument('--exp_name', type=str, required=True)
    create_parser.add_argument('--val_size', type=float, required=True)
    create_parser.add_argument('--basedir', type=str, required=True)
    create_parser.set_defaults(func=create_mixed_exp)

    # synth_dataset_parser = subparsers.add_parser('synth', help='Creates annotations for synth dataset')
    # synth_dataset_parser.add_argument('--img_dir', required=True)
    # synth_dataset_parser.add_argument('--xml_dir', required=True)
    # synth_dataset_parser.add_argument('--output_dir', required=True)
    # create_parser.set_defaults(func=generate_for_synth)
    #
    # synth_dataset_parser = subparsers.add_parser('flir', help='Creates annotations for flir dataset')
    # synth_dataset_parser.add_argument('--img_dir', required=True)
    # synth_dataset_parser.add_argument('--json_dir', required=True)
    # synth_dataset_parser.add_argument('--output_dir', required=True)
    # create_parser.set_defaults(func=generate_for_flir)

    args = parser.parse_args()
    args.func(args)
