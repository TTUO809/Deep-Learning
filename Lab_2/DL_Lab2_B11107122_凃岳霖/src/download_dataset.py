import os
import urllib.request
import zipfile
import tarfile

def download_and_extract(url, extract_to):
    '''
    Args:
        url (str): 下載連結，應該指向一個壓縮檔案（如 .zip 或 .tar.gz）。
        extract_to (str): 解壓縮後的目標資料夾路徑。
    '''
    file_name = url.split('/')[-1]
    file_path = os.path.join(extract_to, file_name)
    folder_name = file_name.rsplit('.', 1)[0]  # 去掉 .zip 或 .tar.gz 等後綴
    target_folder = os.path.join(extract_to, folder_name)

    # 下載檔案
    if not os.path.exists(file_path):
        print(f"Downloading 【{file_name}】 from 【{url}】...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded  【{file_name}】  to  【{file_path}】.")
    else:
        print(f"File 【{file_name}】 already exists at 【{file_path}】. Skipping download.")

    # 解壓縮檔案
    if not os.path.exists(target_folder) or len(os.listdir(target_folder)) == 0:
        print(f"Extracting  【{file_name}】  to  【{target_folder}】...")
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_name.endswith('.tar.gz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported file format for 【{file_name}】. Skipping extraction.")
            return
        print(f"Extracted  【{file_name}】  to  【{target_folder}】.")
    else:
        print(f"Folder 【{target_folder}】 already exists. Skipping extraction.")
    
    print("="*80 + "\n")

def setup_dataset():
    # 取得上一層資料夾的絕對路徑。
    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATASET_DIR = os.path.join(PARENT_DIR, 'dataset')
    PET_DIR = os.path.join(DATASET_DIR, 'oxford_iiit_pets')
    ANNOTATIONS_DIR = os.path.join(PET_DIR, 'annotations')

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    print(f"Dataset will be set up at: {PET_DIR}")

    # Oxford-IIIT Pet Dataset 的下載連結。
    print(f"=== Downloading Oxford-IIIT Pet Dataset... ===")
    official_urls = [
        "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz",
        "https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz"
    ]
    for url in official_urls:
        download_and_extract(url, PET_DIR)

    # Kaggle 競賽專屬資料切分檔案。
    print("=== Decompressing Kaggle Competition Dataset... ===")
    kaggle_zips = [
        "nycu-2026-spring-dl-lab2-unet.zip",
        "binary-semantic-segmentation-res-net-34-u-net.zip"
    ]

    for zip_name in kaggle_zips:
        zip_path = os.path.join(DATASET_DIR, zip_name)
        if not os.path.exists(zip_path):
            print(f"Error: Expected zip file {zip_path} not found. Please ensure you have downloaded it from Kaggle and placed it in the dataset directory.")
            continue

        print(f"Extracting 【{zip_name}】 to 【{ANNOTATIONS_DIR}】...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.endswith('.txt'):
                    filename = os.path.basename(member)
                    if filename:
                        target_path = os.path.join(ANNOTATIONS_DIR, filename)
                        with open(target_path, "wb") as f:
                                f.write(zip_ref.read(member))
        print(f"Extracted .txt files from 【{zip_name}】 to 【{ANNOTATIONS_DIR}】.\n")
    else:
        print("All Kaggle zip files already exist. Skipping extraction.")  

    print("\n" + "="*80)
    print("Dataset setup completed! You can now run train.py to start training your model.")
    print("="*80)

if __name__ == "__main__":
    setup_dataset()