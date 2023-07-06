import os
import requests
import zipfile
from typing import List
from tqdm import tqdm


def download_files(links: List[str], output_dir: str) -> None:
    """Download files to a directory.

    Args:
        links: A list of URLs to download.
        output_dir: A directory to download the files to.
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    # Download each link
    for link in tqdm(links, desc='Downloading files', mininterval=5):
        # Create a filename from the link
        base_name = link.split('?')[0]
        filename = os.path.join(output_dir, os.path.basename(base_name))
        # Download the file
        with requests.get(link, stream=True) as response:
            # Raise an exception if the status code is not 200
            response.raise_for_status()
            # Write the file to disk
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        # If the file is a zip file, extract it
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filename, 'r') as zip_file:
                zip_file.extractall(output_dir)
            # Delete the zip file
            os.remove(filename)


if __name__ == "__main__":
    links = [
        f"https://zenodo.org/record/7908728/files/{x}.zip?download=1"
        for x in range(1, 2)
    ] + ["https://zenodo.org/record/7908728/files/L1C_files.json?download=1"]
    
    
    download_files(links, "./data")
