import cv2
import os

def sliding_window(image, stepSize, windowSize):
    # Slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # Yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def preprocess_for_classification(sub_image, target_size=(224, 224)):
    resized = cv2.resize(sub_image, target_size, interpolation=cv2.INTER_AREA)

    return resized

def read_coordinates(xyc_path):
    coordinates = []
    with open(xyc_path, 'r') as file:
        for line in file:
            x, y = map(int, line.split('\t'))
            coordinates.append((x, y))
    return coordinates

def is_cancerous_window(x, y, windowSize, coordinates):
    for cx, cy in coordinates:
        if x <= cx <= x + windowSize[0] and y <= cy <= y + windowSize[1]:
            return True
    return False

# Parameters
winW, winH = 112, 112
stepSize = 50

# Directory to save images
target_dir = 'Datasets/dataset2'

# Path to the datasets
image_dataset_path = 'Datasets/ALL_IDB1/im/'
coordinates_dataset_path = 'Datasets/ALL_IDB1/xyc/'

# Loop through the dataset
for filename in os.listdir(image_dataset_path):
    if filename.endswith("1.jpg"):
        image_path = os.path.join(image_dataset_path, filename)
        xyc_path = os.path.join(coordinates_dataset_path, filename.replace('.jpg', '.xyc'))
        
        image = cv2.imread(image_path)
        coordinates = read_coordinates(xyc_path)

        cancerous_dir = os.path.join(target_dir, 'all')
        non_cancerous_dir = os.path.join(target_dir, 'hem')
        os.makedirs(cancerous_dir, exist_ok=True)
        os.makedirs(non_cancerous_dir, exist_ok=True)

        img_counter = 0

        for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            processed_window = preprocess_for_classification(window)

            if is_cancerous_window(x, y, (winW, winH), coordinates):
                save_path = os.path.join(cancerous_dir, f"{filename[:-4]}_window_{img_counter}.jpg")
            else:
                save_path = os.path.join(non_cancerous_dir, f"{filename[:-4]}_window_{img_counter}.jpg")

            cv2.imwrite(save_path, processed_window)
            img_counter += 1
