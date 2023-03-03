import os
from concurrent.futures import ThreadPoolExecutor
import humanize
from multiprocessing import Pool
from PIL import Image
import concurrent.futures

"""Delete file function :"""

def delete_file(file):
    try:
        os.remove(file)
    except Exception as e:
        print(f"Error deleting file {file}: {e}")

"""Deleting XML files :"""

def delete_xml_files(directory, max_threads=10):
    # Recursively search for all .xml files in the specified directory and its subdirectories
    xml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, file))

    # Use a thread pool to delete the files with a maximum number of threads
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for file in xml_files:
            executor.submit(delete_file, file)

    print(f"All {len(xml_files)} .xml files have been deleted from {directory} and its subdirectories.")
    # Recursively delete files in subdirectories
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            delete_xml_files(os.path.join(root, dir), max_threads)

"""Deleting unwanted images :"""

def delete_unwanted_images(directory, max_threads=10):
    # Recursively search for all non-xml files in the specified directory and its subdirectories
    view_files = []
    stats_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "VIEW-1_DIFFUSE.JPG" in file:
                view_files.append(os.path.join(root, file))
            elif "STATS.JPG" in file:
                stats_files.append(os.path.join(root, file))

    # Use a thread pool to delete the files with a maximum number of threads
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for file in view_files:
            executor.submit(delete_file, file)
        for file in stats_files:
            executor.submit(delete_file,file)

    print(f"All {len(view_files)} files containing 'VIEW-1_DIFFUSE.JPG' and {len(stats_files)} files containing 'STATS.JPG' in their name have been deleted from {directory} and its subdirectories.")
    
    # Recursively delete files in subdirectories
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            delete_unwanted_images(os.path.join(root, dir), max_threads)



"""Cheking Number of images :"""

def calculate_stats(directory):
    def get_file_paths(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".JPG"):
                    yield os.path.join(root, file)

    images_count = 0
    total_size = 0

    # Use a generator to avoid loading all file paths into memory at once
    file_paths = get_file_paths(directory)

    # Use multiprocessing to parallelize the calculation of file sizes
    with Pool() as pool:
        sizes = pool.map(os.path.getsize, file_paths)
        total_size = sum(sizes)
        images_count = len(sizes)

    # Use humanize to format the total size in a human-readable format
    total_size_humanized = humanize.naturalsize(total_size, binary=True)

    print(f"Number of images: {images_count}")
    print(f"Total size of directory {directory}: {total_size_humanized}")

"""Cheking if the size of the images is the same :"""

def check_images_size(directory):
    sizes = {}
    image_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".jpg"):
                image_paths.append(os.path.join(root, file))

    def get_image_size(path):
        with Image.open(path) as img:
            return img.size

    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(get_image_size, path): path for path in image_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                size = future.result()
                sizes[path] = size
            except Exception as e:
                print(f"Error getting size for image {path}: {e}")

    # Check if all image sizes are the same
    sizes_set = set(sizes.values())
    if len(sizes_set) == 1:
        print(f"All images in {directory} have size {sizes_set.pop()}")
    else:
        print(f"Images in {directory} have different sizes: {sizes}")