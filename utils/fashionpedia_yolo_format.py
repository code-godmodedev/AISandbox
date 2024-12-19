from torch.utils import data
from pathlib import Path
import numpy as np
# from PIL import Image
import yaml
from functools import reduce

class YOLODataset:
    def __init__(self, root_path: Path, train_path: Path, val_path: Path, test_path: Path, class_names: dict) -> None:
        self.path = root_path
        self.train = train_path
        self.val = val_path
        self.test = test_path
        self.names = class_names
    def dump_to_yaml(self) -> Path:
        yamlfile = self.path/Path('data.yaml')

        # Convert object to dictionary
        yolo_dict = {
            "path": str(self.path),
            "train": str(self.train),
            "val": str(self.val),
            "test": str(self.test),
            "names": self.names
        }

        with open(yamlfile, 'w') as yml_file:
            yaml.dump(yolo_dict, yml_file)
        
        return yamlfile

        

class FashionpediaToYOLOFormat:
    def __init__(self, im_save_path=Path('~/.cache/my_datasets/fashionpedia').expanduser()):
        self.requires_sets = ['train', 'val', 'test']
        self.im_save_path = Path(im_save_path).expanduser()
        self.im_save_path.mkdir(parents=True, exist_ok=True)
        # im_paths, _, class_names = self.preprocess()

    def convert_to_yolo_format(self, dataset): 
        
        train_set = dataset['train']
        
        class_names = train_set.features['objects'].feature['category'].names
        class_names = {id: value for id, value in enumerate(class_names)}

        new_split = train_set.train_test_split(test_size=0.2, seed=55)

        dataset_dict = dict(
            train=new_split['train'],
            val=dataset['val'],
            test=new_split['test']
        )
        total_images = reduce(lambda v, y: v+len(y), dataset_dict.values(), 0)

        im_paths=dict(
            train=Path('images/train'),
            val=Path('images/val'),
            test=Path('images/test')
        )
        lbl_paths=dict(
            train=Path('labels/train'),
            val=Path('labels/val'),
            test=Path('labels/test')
        )
        total_processed = 0

        yaml_path = self.im_save_path/Path('data.yaml')
        if not yaml_path.is_file():
            for path, set in dataset_dict.items():

                im_save_dir = self.im_save_path/im_paths[path]
                im_save_dir.mkdir(parents=True, exist_ok=True)

                lbl_save_dir = self.im_save_path/lbl_paths[path]
                lbl_save_dir.mkdir(parents=True, exist_ok=True)
                processed_images = 0
                for item in set:
                    processed_images += 1
                    total_processed += 1
                    if processed_images % 500 == 0:
                        print(f'processing {path} images {processed_images}/{len(set)}, total processed {total_processed}/{total_images}')
                    height = item['height']
                    width = item["width"]
                    image_id = item["image_id"]
                    image = item["image"]

                    image_name = Path(f'{image_id}.jpg')
                    image.save(
                        im_save_dir/image_name,
                        format="JPEG"
                    )

                    objects = item['objects']
                    categories = objects['category']
                    bboxes = np.array(objects['bbox'])

                    bboxes_factor = np.array([[1/width, 1/height, 1/width, 1/height]])
                    bboxes *= bboxes_factor 

                    bboxes_xywh = []

                    for i, category in enumerate(categories):
                        current_bbox = bboxes[i]
                        w = current_bbox[2] - current_bbox[0]
                        h = current_bbox[3] - current_bbox[1]
                        x = (current_bbox[2] + current_bbox[0]) / 2.
                        y = (current_bbox[3] + current_bbox[1]) / 2.
                        bboxes_xywh.append(dict(
                            c=category,
                            x=x,
                            y=y,
                            w=w,
                            h=h
                        ))
                    
                    label_name = Path(f'{image_id}.txt')
                    label_file = lbl_save_dir/label_name

                    # label_file.mkdir(parents=True, exist_ok=True)

                    with open(label_file, "w") as file:
                        for item in bboxes_xywh:
                            file.write(f"{item['c']} {item['x']} {item['y']} {item['w']} {item['h']}\n")
        else:
            print(f'{total_images} already processed')        
        self.processedYOLODataset = YOLODataset(
            root_path=self.im_save_path,
            train_path=im_paths['train'],
            val_path=im_paths['val'],
            test_path=im_paths['test'],
            class_names=class_names,
        )

        return self.processedYOLODataset
                


                            

    
