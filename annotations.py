"""
Generates annottaion for FLIR and improved dataset
"""
import json
import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

import cv2

label_map = {
    '1': 'person',
    '2': 'bicycle',
    '3': 'car'
}

fake_label_map = {
    'Car': 'car',
    'Person': 'person'
}

def process_real_imgs(config):
    json_files = [f for f in os.listdir(config.real_labels_dir) if f.endswith('.json')]
    result = []
    for i, json_file in enumerate(json_files):
        # if i >= config.n_real_imgs:
        #    break
        f = open(os.path.join(config.real_labels_dir, json_file), 'r')
        parsed_json = json.load(f)
        annotations = parsed_json['annotation']
        if len(annotations) == 0:
            result.append({'path': parsed_json['image']['file_name'] + '.jpeg'})
        else:
            for a in annotations:
                if a['category_id'] not in ['1', '2', '3']:
                    continue
                x1 = int(a['bbox'][0])
                x2 = x1 + int(a['bbox'][2])
                y1 = int(a['bbox'][1])
                y2 = y1 + int(a['bbox'][3])
                result.append(
                    {'path': parsed_json['image']['file_name'] + '.jpeg', 'x1': x1,
                     'x2': x2, 'y1': y1, 'y2': y2, 'category': label_map[a['category_id']]})
    return result


def process_fake_dataset(folder, name, config, processed):
    dataset_annotations = get_annotations_xml_from_folder(folder)
    img_dir = os.path.join(folder, name + '_insert')
    results = []
    for i, img_name in enumerate(os.listdir(img_dir)):
        # if processed + i >= config.n_fake_imgs:
        #    break
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
                    results.append({'path': img_name})
                else:
                    results.append({'path': img_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
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
            return 'car'
        else:
            raise Exception('Unrecognized label')
    return fake_label_map[label_attrib]


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
    #real_annot = process_real_imgs(config)
    fake_annot = process_fake_imgs(config)
    # with open('flir.csv', 'w') as f:
    #     for e in real_annot:
    #         f.write(os.path.join(config.final_real_path, e['path'])
    #                 + ',' + str(e.get('x1', '')) + ',' + str(e.get('y1', '')) + ',' + str(
    #                     e.get('x2', '')) + ',' + str(e.get('y2', '')) + ',' + str(e.get('category', '')) + '\n')
    with open('improved_both.csv', 'w') as f:
        for e in fake_annot:
            f.write(os.path.join(config.final_fake_path, e['path'])
                    + ',' + str(e.get('x1', '')) + ',' + str(e.get('y1', '')) + ',' + str(
                        e.get('x2', '')) + ',' + str(e.get('y2', '')) + ',' + str(e.get('category', '')) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract annotations for RetinaNET")
    subparsers = parser.add_subparsers()

    create_parser = subparsers.add_parser('extract')
    create_parser.add_argument('--real_img_dir', type=str, required=True)
    create_parser.add_argument('--real_labels_dir', type=str, required=True)
    create_parser.add_argument('--fake_dir', type=str, required=True, help='Contains both real and fake')
    create_parser.add_argument('--final_real_path', required=True,
                               default='/equilibrium/sgori/dataset/FLIR_Dataset/FLIR_Dataset/training/Data')
    create_parser.add_argument('--final_fake_path',
                               default='/equilibrium/sgori/dataset/improved_dataset_annotations/')
    create_parser.set_defaults(func=create_mixed_exp)

    args = parser.parse_args()
    args.func(args)
