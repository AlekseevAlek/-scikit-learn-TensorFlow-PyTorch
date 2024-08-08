# Разархивирование датасета
import zipfile
import os
# zip_file_path = "C:/Users/raveg/Downloads/archive (1).zip"
# extract_to_path = "C:/Users/raveg/PycharmProjects/pythonProject29/img"
# os.makedirs(extract_to_path, exist_ok=True)
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_to_path)
# extracted_files = os.listdir(extract_to_path)
# print("Извлеченные файлы:")


zip_file_path = "C:/Users/raveg/Downloads/archive.zip"
extract_to_path = "C:/Users/raveg/PycharmProjects/pythonProject29/dataset2"
os.makedirs(extract_to_path, exist_ok=True)
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)
extracted_files = os.listdir(extract_to_path)
print("Извлеченные файлы:")
