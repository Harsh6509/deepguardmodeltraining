import os
import shutil
#useless h ye code , i only used it to fix the structure of the dataset, now i have fixed it so i can delete this file , u can use this to fix the structure of the dataset if u want to use it for training or testing, but i have already fixed it so u can delete this file if u want to
base_dir = r"C:\Users\Raoha\Desktop\capstone\dataset"

# New structure
train_real = os.path.join(base_dir, "train", "real")
train_fake = os.path.join(base_dir, "train", "ai_generated")
test_real = os.path.join(base_dir, "test", "real")
test_fake = os.path.join(base_dir, "test", "ai_generated")

os.makedirs(train_real, exist_ok=True)
os.makedirs(train_fake, exist_ok=True)
os.makedirs(test_real, exist_ok=True)
os.makedirs(test_fake, exist_ok=True)

# Move files
def move_files(src, dest):
    for file in os.listdir(src):
        shutil.move(os.path.join(src, file), os.path.join(dest, file))

move_files(os.path.join(base_dir, "real", "train"), train_real)
move_files(os.path.join(base_dir, "real", "test"), test_real)
move_files(os.path.join(base_dir, "ai_generated", "train"), train_fake)
move_files(os.path.join(base_dir, "ai_generated", "test"), test_fake)

print("Structure fixed ✅")