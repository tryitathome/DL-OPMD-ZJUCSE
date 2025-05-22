import os
import numpy as np
from PIL import Image
from tqdm import tqdm


'''建立memmap文件'''
def create_memmap_file(dataset_dir, output_file, img_shape, description):
    if os.path.exists(output_file):
        return
    
    class_to_idx = {}
    data = []
    
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
            for img_name in tqdm(os.listdir(class_path), desc=f'Processing {description} images in {class_name}', leave=True):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('RGB').resize((img_shape[1], img_shape[2]), Image.Resampling.LANCZOS)

                img_array = np.array(img, dtype=np.uint8).transpose((2, 0, 1))
                data.append((img_array, class_to_idx[class_name]))
    
    num_images = len(data)
    dtypes = np.dtype([('image', np.uint8, img_shape), ('label', np.int64)])
    mmap = np.memmap(output_file, dtype=dtypes, mode='w+', shape=(num_images,))
    
    for i, (image, label) in enumerate(tqdm(data, desc='Saving to memmap')):
        mmap[i]['image'] = image
        mmap[i]['label'] = label
    mmap.flush()