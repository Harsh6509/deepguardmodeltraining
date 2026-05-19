from PIL import Image
import os
from tqdm import tqdm
#ye wali file ka use maine images ko resize karne ke liye kiya tha, abhi maine images ko resize kar diya hai to ye file bhi useless ho gayi hai, agar tumhe images ko resize karna hai to tum is file ka use kar sakte ho, lekin maine already images ko resize kar diya hai to tum is file ko delete kar sakte ho, agar tumhe zarurat nahi hai to
# yaha pe dataset address dalna hai 
input_dirs = [
    r"C:\Users\Raoha\Desktop\capstone\dataset\ai_generated",
    r"C:\Users\Raoha\Desktop\capstone\dataset\real"
]
#yaha dhyan rakhna hai ki image size ko 224x224 karna hai, agar tumhe 224x224 se alag size chahiye to tum SIZE variable me change kar sakte ho, lekin maine 224x224 set kiya hai to tum bhi 224x224 hi use karo, 
# Resize target
SIZE = (224, 224)   # change to (224,224) if needed

for folder in input_dirs:
    print(f"Processing: {folder}")
    
    for filename in tqdm(os.listdir(folder)):
        path = os.path.join(folder, filename)

        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(SIZE)
            img.save(path, "JPEG", quality=90)
        except Exception as e:
            print(f"Skipped: {filename}")

print("✅ All images resized successfully!")