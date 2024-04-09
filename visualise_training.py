import matplotlib.pyplot as plt

# Load validation accuracy and loss
val_loss = []
val_accuracy = []

with open('training_valloss_valaccuracy.json', 'r') as file:
    for line in file:
        data = json.loads(line)
        val_loss.append(data['val_loss'])
        val_accuracy.append(data['val_accuracy'])


# Create an array of epoch numbers for the x-axis
epochs = [i for i in range(1, len(val_accuracy)+1)]

plt.figure(figsize=(8, 6))

# Plot accuracy
accuracy_line, = plt.plot(epochs, val_accuracy, marker='o', linestyle='-', color='b', label='Validation Accuracy')

# Plot loss on the same graph
loss_line, = plt.plot(epochs, val_loss, marker='x', linestyle='--', color='r', label='Validation Loss')

# Set the limits of the y-axis for accuracy
plt.ylim(0, 1)

# Create legend
plt.legend(handles=[accuracy_line, loss_line], loc='best')

# Add title and labels
plt.title("Accuracy and Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Values")

# Add grid
plt.grid(True)

# Save the plot to a file
plt.savefig("Metrics/Accuracy and Loss vs epoch.png")
