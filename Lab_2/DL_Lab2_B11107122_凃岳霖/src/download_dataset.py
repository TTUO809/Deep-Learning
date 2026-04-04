import os
import urllib.request
import zipfile
import tarfile

def _infer_target_folder_name(file_name):
    """
    Args:
        file_name (str): 壓縮檔案的名稱。
    Returns:
        (str): 從檔案名稱中推斷出的資料夾名稱。
    Description:
        根據壓縮檔案的名稱推斷出解壓縮後的資料夾名稱。這個函數會檢查檔案名稱的副檔名，並根據常見的壓縮格式（如 .tar.gz 和 .zip ）
        來去掉相應的副檔名，從而得到資料夾名稱。
    """

    if file_name.endswith('.tar.gz'):
        return file_name[:-7]
    if file_name.endswith('.zip'):
        return file_name[:-4]
    return file_name.rsplit('.', 1)[0]

def _safe_extract_tar(tar_ref, extract_to):
    """
    Args:
        tar_ref (tarfile.TarFile): 已打開的 tar 檔案對象。
        extract_to (str): 解壓縮的目標資料夾路徑。
    Description:
        安全地解壓縮 tar 檔案，防止路徑穿越攻擊。這個函數會檢查 tar 檔案中的每個成員的路徑，
        確保它們都在指定的解壓縮目錄下，從而避免惡意檔案被解壓縮到不安全的位置。
    """
    
    base_dir = os.path.abspath(extract_to)
    for member in tar_ref.getmembers():
        target_path = os.path.abspath(os.path.join(extract_to, member.name))
        if not target_path.startswith(base_dir + os.sep) and target_path != base_dir:
            raise ValueError(f"Unsafe path detected in tar file: {member.name}")
    tar_ref.extractall(extract_to)

def _download_and_extract(url, extract_to):
    """
    Args:
        url (str): 下載連結，應該指向一個壓縮檔案（如 .zip 或 .tar.gz）。
        extract_to (str): 解壓縮後的目標資料夾路徑。
    Description:
        1. 從指定的 URL 下載壓縮檔案，並將其保存到 extract_to 目錄下。
        2. 根據檔案的副檔名判斷使用哪種解壓縮方法（zipfile 或 tarfile）來解壓縮檔案。
        3. 解壓縮後的資料夾名稱是從檔案名稱中去掉副檔名得到的，並將解壓縮的內容放在 extract_to 目錄下。
        4. 在下載和解壓縮過程中，會檢查檔案是否已經存在，以避免重複下載和解壓縮。
    """

    file_name = url.split('/')[-1]
    file_path = os.path.join(extract_to, file_name)
    folder_name = _infer_target_folder_name(file_name)
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
                _safe_extract_tar(tar_ref, extract_to)
        else:
            print(f"Unsupported file format for 【{file_name}】. Skipping extraction.")
            return
        print(f"Extracted   【{file_name}】 !!!!")
    else:
        print(f"Folder 【{target_folder}】 already exists. Skipping extraction.")
    
    print()

def setup_dataset():
    """
    Description:
        這個函數負責下載和設置 Oxford-IIIT Pet Dataset ，並將 Kaggle 競賽專屬資料切分檔案解壓縮到指定的資料夾中。
        1. 使用 _download_and_extract 函數從官方提供的 URL 下載圖像和標註資料，並將其解壓縮到 DATA_DIR 中。
        2. 接著，從 Kaggle 競賽專屬資料中提取包含切分資訊的 .txt 檔案，並將它們放在 ANNOTATIONS_DIR 中，以供後續的數據加載和處理使用。
        3. 在整個過程中，函數會檢查檔案和資料夾是否已經存在，以避免重複下載和解壓縮，確保效率和資源的合理使用。
    """

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    print(f"Dataset will be set up at: {DATA_DIR}")

    # 下載 Oxford-IIIT Pet Dataset 官方提供的圖像和標註資料。
    print(f"===== Downloading Oxford-IIIT Pet Dataset... =====")
    official_urls = [
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    ]
    for url in official_urls:
        _download_and_extract(url, DATA_DIR)

    # 解壓縮 Kaggle 競賽專屬資料切分檔案。
    print("===== Decompressing Kaggle Competition Dataset... =====")
    kaggle_zips = [
        "nycu-2026-spring-dl-lab2-unet.zip",
        "nycu-2026-spring-dl-lab2-res-net-34-unet.zip"
    ]

    # 檢查 Kaggle 競賽專屬資料的 .zip 檔案是否存在，並將其中的 .txt 檔案解壓縮到 ANNOTATIONS_DIR 中。
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
                        if os.path.exists(target_path):
                            print(f"Skip existing txt: 【{target_path}】")
                            continue
                        with open(target_path, "wb") as f:
                            f.write(zip_ref.read(member))
        print(f"Extracted  【{zip_name}】 !!!!\n")

    print("="*60)
    print("Dataset setup completed! You can now run train.py to start training your model.")
    print("="*60)

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))    # path-to-/src/
    PROJECT_DIR = os.path.dirname(CURRENT_DIR)                  # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/
    DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')          # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/dataset
    DATA_DIR = os.path.join(DATASET_DIR, 'oxford-iiit-pet')     # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/dataset/oxford-iiit-pet
    ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations')     # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/dataset/oxford-iiit-pet/annotations

    setup_dataset()