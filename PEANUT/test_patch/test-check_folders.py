import os

def check_folders(parent_folder):
    folders = os.listdir(parent_folder)
    for folder in folders:
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                if '499' in file:
                    print(folder)
                    break


parent_folder = ""


check_folders(parent_folder)
