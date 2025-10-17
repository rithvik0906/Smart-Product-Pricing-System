# download_images.py
import pandas as pd
import argparse
from src.utils import download_images
import os

def main(limit):
    print("🚀 Starting image download...")
    df = pd.read_csv("student_resource/dataset/train.csv")
    # if you already have 2000 downloaded, we will pick the next 3000 unique links
    start = 0
    # to be safe, list files already in images folder
    image_folder = "student_resource/images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    existing_files = set(os.listdir(image_folder))
    links = []
    for link in df['image_link'].tolist():
        if not isinstance(link, str) or link.strip()=="":
            continue
        fname = link.split("/")[-1]
        if fname in existing_files:
            continue
        links.append(link)
        if len(links) >= limit:
            break

    print(f"Will download {len(links)} new images to {image_folder}")
    if len(links) > 0:
        download_images(links, image_folder)
    print("✅ Images downloaded successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3000, help="how many new images to download")
    args = parser.parse_args()
    main(args.limit)
