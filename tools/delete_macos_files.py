import os

def delete_mac_hidden_files(directory):
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.startswith('.'):
                file_path = os.path.join(root, name)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    target_directory = 'data'
    delete_mac_hidden_files(target_directory)
