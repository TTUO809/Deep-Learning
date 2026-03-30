import os
import urllib.request
import zipfile
import tarfile
import subprocess

def download_and_extract(url, extract_to):
    '''
    Args:
        url (str): 下載連結，應該指向一個壓縮檔案（如 .zip 或 .tar.gz）。
        extract_to (str): 解壓縮後的目標資料夾路徑。
    '''
    # 確保目標資料夾存在。
    os.makedirs(extract_to, exist_ok=True)
    filename = url.split('/')[-1]
    file_path = os.path.join(extract_to, filename)

    # 下載檔案
    if not os.path.exists(file_path):
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {filename} to {file_path}")
    else:
        print(f"File {filename} already exists at {file_path}. Skipping download.")

    # 解壓縮檔案
    folder_name = filename.rsplit('.', 1)[0]  # 去掉 .zip 或 .tar.gz 等後綴
    target_folder = os.path.join(extract_to, folder_name)

    # 檢查目標資料夾是否已經存在且非空，如果是則跳過解壓縮。
    need_extract = False
    if not os.path.exists(target_folder):
        need_extract = True
    elif len(os.listdir(target_folder)) == 0:
        print(f"Folder {target_folder} exists but is empty. Will re-extract.")
        need_extract = True

    if need_extract:
        print(f"Extracting {filename} to {extract_to}...")
        if filename.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif filename.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        print(f"Completed extraction!!!")
    else:
        print(f"Folder {target_folder} already exists and is not empty. Skipping extraction.")

def setup_dataset():
    # 取得上一層資料夾的絕對路徑。
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dataset_dir = os.path.join(parent_dir, 'dataset', 'oxford-iiit-pet')
    annotations_dir = os.path.join(dataset_dir, 'annotations')

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    print(f"Dataset will be set up at: {dataset_dir}")

    # Oxford-IIIT Pet Dataset 的下載連結。
    print(f"=== Downloading Oxford-IIIT Pet Dataset... ===")
    image_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    masks_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    download_and_extract(image_url, extract_to=dataset_dir)
    download_and_extract(masks_url, extract_to=dataset_dir)

    # Kaggle 競賽專屬資料切分檔案。
    print("=== Downloading Kaggle-specific dataset... ===")
    competition_url = [
        "nycu-2026-spring-dl-lab2-unet",
        "binary-semantic-segmentation-res-net-34-u-net"
    ]

    for comp in competition_url:
        print(f"--- Downloading dataset for {comp} ---")
        zip_path = os.path.join(dataset_dir, f"{comp}.zip")

        if not os.path.exists(zip_path):
            result = subprocess.run(["kaggle", "competitions", "download", "-c", comp, "-p", dataset_dir], capture_output=True, text=True)
            
            if "401 - Unauthorized" in result.stderr or "403 - Forbidden" in result.stderr:
                print(f"Error: Authentication failed when trying to download {comp}. Please ensure you have set up Kaggle API credentials correctly.")
                continue
            elif result.returncode != 0:
                print(f"Error: Failed to download {comp}. Kaggle API returned an error: {result.stderr}")
        else:
            print(f"File {zip_path} already exists. Skipping Kaggle download.")
        
        if os.path.exists(zip_path):
            extracted_txt_count = 0
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith('.txt'):
                        file_name_only = os.path.basename(file_info.filename)
                        if file_name_only:
                            target_path = os.path.join(annotations_dir, file_name_only)
                            if not os.path.exists(target_path):
                                file_info.filename = file_name_only
                                zip_ref.extract(file_info, annotations_dir)
                                extracted_txt_count += 1
                                print(f"Extracted {file_name_only} to {annotations_dir}")
        
            if extracted_txt_count == 0:
                print(f"Warning: No .txt files were extracted from {zip_path}. Please check the contents of the zip file.")

            os.remove(zip_path)  # 刪除 zip 檔案以節省空間。
            print(f"Removed zip file: {zip_path}")

    print("\n" + "="*60)
    print("Dataset setup completed! You can now run train.py to start training your model.")
    print("="*60)

if __name__ == "__main__":
    setup_dataset()