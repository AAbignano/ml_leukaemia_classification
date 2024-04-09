import os
import shutil
import csv
import zipfile

# Set file paths
zip_file   = 'Datasets/C-NMC_Leukemia.zip'
target_dir = 'Datasets/dataset1'

# Create the target directory
os.makedirs(target_dir, exist_ok=True)

# Folders to extract
folders_to_extract = ['C-NMC_training_data', 'C-NMC_test_prelim_phase_data']

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    # List of all archived file names from the zip
    all_files = zip_ref.namelist()

    # Filter the files to extract
    files_to_extract = [f for f in all_files if any(folder in f for folder in folders_to_extract)]

    # Extract the selected files
    for file in files_to_extract:
        zip_ref.extract(file, target_dir)

# Folders to combine
folders_to_combine = ['all', 'hem']

# Create new folders for combined data
combined_dirs = {}
for folder in folders_to_combine:
    combined_dir = os.path.join(target_dir, folder)
    os.makedirs(combined_dir, exist_ok=True)
    combined_dirs[folder] = combined_dir

# Copy and combine contents
for fold in ['fold_0', 'fold_1', 'fold_2']:
    for folder in folders_to_combine:
        src_dir = os.path.join(target_dir, 'C-NMC_Leukemia', folders_to_extract[0], fold, folder)

        for filename in os.listdir(src_dir):

            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(combined_dirs[folder], filename)

            shutil.move(src_file, dst_file)

# Process test data based on CSV
csv_file = os.path.join(target_dir, 'C-NMC_Leukemia', folders_to_extract[1], 'C-NMC_test_prelim_phase_data_labels.csv')
with open(csv_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')

    # Skip header row
    next(csv_reader)

    for row in csv_reader:
        patient_id, filename, label = row

        src_file = os.path.join(target_dir, 'C-NMC_Leukemia', folders_to_extract[1], filename)
        dst_folder = 'hem' if label == '0' else 'all'
        dst_file = os.path.join(combined_dirs[dst_folder], patient_id)

        shutil.move(src_file, dst_file)

# Remove remaining empty folder
old_dir = os.path.join(target_dir, 'C-NMC_Leukemia')
shutil.rmtree(old_dir)
