import json
import os
import cv2
import glob
import argparse
import shutil


def process_dataset(input_json_paths, output_folder, id_prefix, fixed_image_path):
    os.makedirs(output_folder, exist_ok=True)
    images_folder = os.path.join(output_folder, "images")
    labels_folder = os.path.join(output_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # 读取固定图片
    fixed_image = cv2.imread(fixed_image_path)
    if fixed_image is None:
        raise ValueError(f"无法读取图片: {fixed_image_path}")

    processed_ids = set()
    id_counter = 0

    for input_json_path in input_json_paths:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            new_id = f"{id_prefix}{id_counter:06d}"
            id_counter += 1

            while new_id in processed_ids:
                new_id = f"{id_prefix}{id_counter:06d}"
                id_counter += 1

            processed_ids.add(new_id)

            conversations = item['conversations']

            # 复制固定图片
            image_path = os.path.join(images_folder, f"{new_id}.png")
            cv2.imwrite(image_path, fixed_image)

            # 创建对应的JSON文件
            label_data = {
                "conversations": [
                                     {
                                         "role": "user",
                                         "content": "<image>\n" + conversations[0]['value']
                                     }
                                 ] + [
                                     {
                                         "role": "assistant" if conv['from'] == "assistant" else "user",
                                         "content": conv['value']
                                     } for conv in conversations[1:]
                                 ]
            }

            label_path = os.path.join(labels_folder, f"{new_id}.json")
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)

    print(f"Dataset processed and saved in {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Process multiple JSON files to create a dataset with a fixed image.")
    parser.add_argument("--input_folder", default="/home/iiap/PycharmProjects/再次开始的deeplearning/util/jsons",
                        help="Folder containing input JSON files (default: ./input)")
    parser.add_argument("--output_folder", default="/home/iiap/桌面/数据集/cogdata_text",
                        help="Folder to save the processed dataset (default: ./output)")
    parser.add_argument("--prefix", default="TEXT_", help="Prefix for new IDs (default: TEXT_)")
    parser.add_argument("--fixed_image", default="/media/iiap/25df545d-3a24-4466-b58d-f96c46b9a3bf/animagine-xl-3.1/output/Ayanami_test.png", help="Path to the fixed image to use for all entries")
    args = parser.parse_args()

    input_json_paths = glob.glob(os.path.join(args.input_folder, "*.json"))

    if not input_json_paths:
        print(f"No JSON files found in {args.input_folder}")
        return

    process_dataset(input_json_paths, args.output_folder, args.prefix, args.fixed_image)


if __name__ == "__main__":
    main()

'''
   parser.add_argument("--input_folder", default="/home/iiap/PycharmProjects/再次开始的deeplearning/util/jsons")
    parser.add_argument("--output_folder", default="/home/iiap/桌面/数据集/cogdata_text")
    parser.add_argument("--prefix", default="TEXT_", help="Prefix for new IDs (default: TEXT_)")
'''