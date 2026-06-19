import os
import shutil
import random

source_dir = "dataset"
target_dir = "dataset_split"

split_ratio = (0.7, 0.15, 0.15)

for category in ["phishing", "legitimate"]:

    path = os.path.join(source_dir, category)
    images = os.listdir(path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for folder in ["train", "validation", "test"]:
        os.makedirs(os.path.join(target_dir, folder, category), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(path, img),
                    os.path.join(target_dir, "train", category, img))

    for img in val_imgs:
        shutil.copy(os.path.join(path, img),
                    os.path.join(target_dir, "validation", category, img))

    for img in test_imgs:
        shutil.copy(os.path.join(path, img),
                    os.path.join(target_dir, "test", category, img))

print("Dataset successfully split!")