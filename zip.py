import zipfile
import os


ziplist = ['lexical_overlap.py',
            'model.py',
            'preprocess.py',
            'rdata.py',
            'train.py',
            'train_bert.py',
            'test_bert.py',
            'training_config.py', 'bert.py', 'w2v.py', 'rddata.py']

def zip_specific_files(dirpath, file_names, ziph, zip_folder_name):
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if file in file_names:
                file_path = os.path.join(root, file)
                arcname = os.path.join(zip_folder_name, os.path.relpath(file_path, dirpath))
                ziph.write(file_path, arcname)


# 指定要压缩的子文件夹及其对应的目标文件夹名称
directories_to_zip = [
    ('taskA', 'Track A'),
    ('taskB', 'Track B')
]

# 创建一个新的压缩文件
with zipfile.ZipFile('2100012909.zip', 'w') as zipf:
    for dirpath, zip_folder_name in directories_to_zip:
        zip_specific_files(dirpath, ziplist, zipf, zip_folder_name)

print("指定文件夹中的指定文件压缩成功！")


