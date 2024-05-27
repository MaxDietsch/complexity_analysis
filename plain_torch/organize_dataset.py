import os
import random
import shutil

def organize_dataset(base_dir, train_pct, val_pct, test_pct):
    class_id = 0
    meta_dir = os.path.join(base_dir, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    train_dir = os.path.join(base_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(base_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    for root, dirs, files in os.walk(base_dir):
        if not files or root in [meta_dir, train_dir, val_dir, test_dir]:
            continue  # Skip empty directories or the meta directory
        

        class_id = int(root[-1])

        total_files = len(files)
        num_train = int(total_files * train_pct / 100)
        num_val = int(total_files * val_pct / 100)
        num_test = total_files - num_train - num_val

        print(f"total files: {total_files}")
        print(f"number train files: {num_train}")
        print(f"number validation files: {num_val}")
        print(f"number test files: {num_test}")

        # Shuffle files for random selection
        random.shuffle(files)

        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:]

        # Copy and Write to files
        copy_and_write(train_files, root, class_id, base_dir, 'train', meta_dir)
        copy_and_write(val_files, root, class_id, base_dir, 'val', meta_dir)
        copy_and_write(test_files, root, class_id, base_dir, 'test', meta_dir)
        print(f"id of {root} is: {class_id}")
        class_id += 1

def copy_and_write(file_list, root, class_id, base_dir, category, meta_dir):
    for file in file_list:
        # Copying file
        source = os.path.join(root, file)
        destination = os.path.join(base_dir, category, file)
        shutil.copy(source, destination)

        # Writing to meta file
        with open(os.path.join(meta_dir, f'{category}.txt'), 'a') as f:
            f.write(f"{destination} {class_id}\n")

# Usage
base_directory = '../dataset_default'
train_percent = 70  
val_percent = 10 
test_percent = 20 

organize_dataset(base_directory, train_percent, val_percent, test_percent)

