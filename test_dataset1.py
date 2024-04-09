# Pytorch
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader

# Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# GradCam++
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# LIME
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm

# SHAP
import shap


#
## Generate Metrics
#


## Load the model

# VGG19
model = models.vgg19(weights=None)

# Add a MaxPooling layer at the top of the model
new_features = torch.nn.Sequential(
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    *model.features
)

# Replace the original features with the new_features
model.features = new_features

num_classes = 1
model.classifier[-1] = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[-1].in_features, num_classes),
        torch.nn.Sigmoid()
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f'Using {device} for inference')

# Load model weights
model.load_state_dict(torch.load("Model_Weights/vgg19.pth"))

# Load the test dataloader
test_dataloader = torch.load("Cache/dataset1_test_dataloader.pth")

# Set number of images displayed in each graph
number_of_visualisations = 3

# Testing phase
model.eval()
all_labels = []
all_predictions = []

correctly_predicted_images = []
incorrectly_predicted_images = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs).squeeze()

        outputs = outputs.cpu()
        labels = labels.cpu()

        predictions = torch.round(outputs)

        all_labels.extend(labels)
        all_predictions.extend(predictions.numpy())

        if len(all_labels) < number_of_visualistions:
            for i in range(len(labels)):
                if labels[i] == predictions[i]:
                    correctly_predicted_images.append(inputs[i].cpu())
                else:
                    incorrectly_predicted_images.append(inputs[i].cpu())


## Calculate Evaluation Metrics

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Set the labels for the heatmap
labels = ["all", "hem"]

# Reshape confusion matrix into a 2x2 matrix
cm_matrix = np.array(cm)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.savefig("Metrics/dataset1 - Confusion Matrix.png")

# Calcuate evaluation metrics
accuracy  =  accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall    =    recall_score(all_labels, all_predictions)
f1        =        f1_score(all_labels, all_predictions)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")




#
## Interpretability
#



# Load dataset1 mean and std
loaded_tensors = torch.load('Cache/dataset1_mean+std.pth')

mean = loaded_tensors['mean'].numpy()
std  = loaded_tensors['std'].numpy()


### GRAD-CAM++
target_layers = [model.features[-1]]

cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

fig, axes = plt.subplots(1, number_of_visualisations, figsize=(15, 5))
    
targets = [ClassifierOutputTarget(0)]

## Correct
for i in range(number_of_visualisations):

    image = correctly_predicted_images[i].permute(1, 2, 0).numpy()
    image = (image * std) + mean
    image = np.clip(image, 0, 1)

    input_tensor = correctly_predicted_images[i]
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    
    axes[i].imshow(visualization)
    axes[i].set_title(f'Correct {i+1}')
    axes[i].axis('off')

    plt.savefig("Metrics/dataset1 - GradCamPP Correct.png")

## Incorrect
for i in range(number_of_visualisations):

    image = incorrectly_predicted_images[i].permute(1, 2, 0).numpy()
    image = (image * std) + mean
    image = np.clip(image, 0, 1)

    input_tensor = incorrectly_predicted_images[i]
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    
    axes[i].imshow(visualization)
    axes[i].set_title(f'Incorrect {i+1}')
    axes[i].axis('off')

    plt.savefig("Metrics/dataset1 - GradCamPP Incorrect.png")




### LIME

# Instantiate LimeImageExplainer
explainer = lime_image.LimeImageExplainer()

def batch_predict(images):
    model.eval()

    batch = torch.stack(tuple(torch.from_numpy(np.transpose(im, (2, 0, 1))) for im in images), dim=0)
    batch = batch.to(device)

    with torch.no_grad():
        outputs = model(batch)

    return outputs.cpu().numpy()

# Create a new figure with subplots
fig, axes = plt.subplots(1, number_of_visualisations, figsize=(20, 4))

# Create a segmentation function
segmentation_fn = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)

## Correct 
for i in range(number_of_visualisations):

    image = correctly_predicted_images[i].permute(1, 2, 0).numpy()
    image = (image * std) + mean
    image = np.clip(image, 0, 1)

    # Explain the prediction of that image using LIME
    explanation = explainer.explain_instance(image,
                                             batch_predict,
                                             top_labels=1,
                                             hide_color=0,
                                             num_samples=1000,
                                             segmentation_fn=segmentation_fn)

    # Get the image and mask back for the top explanation
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

    # Plot the explanation on the i-th subplot
    axes[i].imshow(mark_boundaries(temp, mask))
    axes[i].axis('off')
    axes[i].set_title(f'Correct {i+1}')

# Adjust the layout
plt.tight_layout()

# Save the plot to a file
plt.savefig("Metrics/dataset1 - LIME Correct.png", bbox_inches='tight')

## Incorrect 
for i in range(number_of_visualisations):

    image = incorrectly_predicted_images[i].permute(1, 2, 0).numpy()
    image = (image * std) + mean
    image = np.clip(image, 0, 1)

    # Explain the prediction of that image using LIME
    explanation = explainer.explain_instance(image,
                                             batch_predict,
                                             top_labels=1,
                                             hide_color=0,
                                             num_samples=1000,
                                             segmentation_fn=segmentation_fn)

    # Get the image and mask back for the top explanation
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

    # Plot the explanation on the i-th subplot
    axes[i].imshow(mark_boundaries(temp, mask))
    axes[i].axis('off') 
    axes[i].set_title(f'Incorrect {i+1}')

# Adjust the layout
plt.tight_layout()

# Save the plot to a file
plt.savefig("Metrics/dataset1 - LIME Incorrect.png", bbox_inches='tight')




### SHAP

## Correct

# 1. Select the first 'number_of_visualisations' images
selected_images = correctly_predicted_images[:number_of_visualisations]

# 2. Create tensors from the selected images
images_tensor = torch.stack(selected_images).to(device)

# 3. Initialize the SHAP explainer

# Create a black image for the background
black_background = torch.zeros_like(images_tensor[0]).unsqueeze(0).to(device)

explainer = shap.GradientExplainer((model, model.features), black_background)

# 4. Compute SHAP values
shap_values = explainer.shap_values(images_tensor)

# 5. Plot the SHAP values
# Convert the tensor images to numpy for plotting
image_numpy = [np.transpose(img.cpu().numpy(), (1, 2, 0)) for img in images_tensor]

# Convert SHAP values for the positive class to a numpy array
shap_values_array = np.array(shap_values)

# Convert the tensor images back to numpy for plotting
image_numpy_plot = []
for image in selected_images:
    image = image.permute(1, 2, 0).numpy()
    image = (image * std) + mean
    image = np.clip(image, 0, 1)

    image_numpy_plot.append(image)

# Set the figure size (width, height) in inches
figure_width = 32  # or any other size you want
figure_height = figure_width * (377.0 / 1234.0)  # Preserve the aspect ratio

# Set the dots per inch (resolution) of the figure
dpi = 100  # This will result in a 1600x488 pixel image, as an example

# Create a new figure with the specified size and dpi
plt.figure(figsize=(figure_width, figure_height), dpi=dpi)

# Set up the subplots
fig, axes = plt.subplots(1, number_of_visualisations, figsize=(15, 5))

for i, ax in enumerate(axes.flat):
    image = image_numpy_plot[i]
    
    # Ensure the SHAP values have the correct shape (height, width, channels)
    shap_value = np.transpose(shap_values_array[i], (1, 2, 0))
    
    # Ensure SHAP values only have height and width dimensions for plotting
    shap_value_summed = np.sum(shap_value, axis=2)
    
    # Display the image
    ax.imshow(image)

    # Overlay the SHAP heatmap
    im = ax.imshow(shap_value_summed, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title(f"Correct {i+1}")

# Add a colorbar to the right of the subplots
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# Save the full figure with all subplots
plt.savefig("Metrics/dataset1 - SHAP Correct.png", bbox_inches='tight')

## Incorrect

# 1. Select the first 'number_of_visualisations' images
selected_images = incorrectly_predicted_images[:number_of_visualisations]

# 2. Create tensors from the selected images
images_tensor = torch.stack(selected_images).to(device)

# 3. Initialize the SHAP explainer

# Create a black image for the background
black_background = torch.zeros_like(images_tensor[0]).unsqueeze(0).to(device)

explainer = shap.GradientExplainer((model, model.features), black_background)

# 4. Compute SHAP values
shap_values = explainer.shap_values(images_tensor)

# 5. Plot the SHAP values
# Convert the tensor images to numpy for plotting
image_numpy = [np.transpose(img.cpu().numpy(), (1, 2, 0)) for img in images_tensor]

# Convert SHAP values for the positive class to a numpy array
shap_values_array = np.array(shap_values)

# Convert the tensor images back to numpy for plotting
image_numpy_plot = []
for image in selected_images:
    image = image.permute(1, 2, 0).numpy()
    image = (image * std) + mean
    image = np.clip(image, 0, 1)

    image_numpy_plot.append(image)

# Set up the subplots
fig, axes = plt.subplots(1, number_of_visualisations, figsize=(15, 5))

for i, ax in enumerate(axes.flat):
    image = image_numpy_plot[i]
    
    # Ensure the SHAP values have the correct shape (height, width, channels)
    shap_value = np.transpose(shap_values_array[i], (1, 2, 0))
    
    # Ensure SHAP values only have height and width dimensions for plotting
    shap_value_summed = np.sum(shap_value, axis=2)
    
    # Display the image
    ax.imshow(image)

    # Overlay the SHAP heatmap
    im = ax.imshow(shap_value_summed, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title(f"Incorrect {i+1}")

# Add a colorbar to the right of the subplots
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# Save the full figure with all subplots
plt.savefig("Metrics/dataset1 - SHAP Incorrect.png", bbox_inches='tight')
