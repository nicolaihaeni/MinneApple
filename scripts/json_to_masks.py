import os
import sys
import cv2
import json
import shutil
import numpy as np


def create_output_folders(out_path, parts):
    for part in parts:
        if not os.path.exists(os.path.join(out_path, part, 'images')):
            os.makedirs(os.path.join(out_path, part, 'images'))
        if not os.path.exists(os.path.join(out_path, part, 'masks')):
            os.makedirs(os.path.join(out_path, part, 'masks'))


def save_output_images(in_path, out_path, part):
    datasets = os.listdir(in_path)
    for dataset in datasets:
        dataset_path = os.path.join(in_path, dataset)
        json_path = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
        annotations = json.load(open(os.path.join(dataset_path, json_path[0])))

        for a in annotations:
            if annotations[a]["regions"]:
                if part == 'train':
                    image_name = annotations[a]["filename"]
                    shutil.copy2(os.path.join(dataset_path, image_name), os.path.join(out_path, "images", dataset + "_" + image_name))
                else:
                    image_name = annotations[a]["filename"]
                    image = cv2.imread(os.path.join(in_path, dataset, image_name))
                    image = cv2.resize(image, (720, 1280))
                    cv2.imwrite(os.path.join(out_path, "images", dataset + "_" + image_name), image)


def save_output_masks(in_path, out_path, part):
    datasets = os.listdir(in_path)
    for dataset in datasets:
        dataset_path = os.path.join(in_path, dataset)
        json_path = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
        annotations = json.load(open(os.path.join(dataset_path, json_path[0])))

        for a in annotations:
            if annotations[a]["regions"]:
                image_name = annotations[a]["filename"]
                mask_path = os.path.join(out_path, "masks", dataset + "_" + image_name)

                # Create the mask file
                mask = np.zeros((1280, 720), np.uint8)
                count = 1
                for region in annotations[a]["regions"]:
                    print(dataset)
                    if region['region_attributes']['class'].lower() == 'Apple'.lower():
                        x = region["shape_attributes"]["all_points_x"]
                        y = region["shape_attributes"]["all_points_y"]

                        pts = np.array(np.column_stack((x,y)), np.int32)
                        pts = pts.reshape((-1,1,2))
                        cv2.fillPoly(mask, [pts], count)
                        count += 1

                cv2.imwrite(mask_path, mask)


def main():
    args = get_args()
    base_path = args.base_path
    out_path = args.out_path
    parts = ['train', 'test']

    #Create output folders
    create_output_folders(out_path, parts)

    for part in parts:
        read_path = os.path.join(base_path, part)
        write_path = os.path.join(out_path, part)

        save_output_images(read_path, write_path, part)
        save_output_masks(read_path, write_path, part)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str)
    parser.add_argument('--out_path', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    main()
