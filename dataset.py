import os
import zipfile
import urllib.request

DATA_DIR = "/datasets/coco"
IMG_DIR = f"{DATA_DIR}/val2017"
ANN_DIR = f"{DATA_DIR}/annotations"

os.makedirs(DATA_DIR, exist_ok=True)

# 1. Download Annotations
if not os.path.exists(f"{ANN_DIR}/instances_val2017.json"):
    print("Downloading Annotations...")
    urllib.request.urlretrieve(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "coco_ann.zip"
    )
    with zipfile.ZipFile("coco_ann.zip", 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove("coco_ann.zip")
    print("Annotations downloaded.")

# 2. Download Validation Images
if not os.path.exists(IMG_DIR):
    print("Downloading Validation Images (this may take a minute)...")
    urllib.request.urlretrieve(
        "http://images.cocodataset.org/zips/val2017.zip",
        "coco_val.zip"
    )
    with zipfile.ZipFile("coco_val.zip", 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove("coco_val.zip")
    print("Images downloaded.")

print(f"Data ready at: {DATA_DIR}")