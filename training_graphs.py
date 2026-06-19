import matplotlib.pyplot as plt

# Example values
# Replace these with YOUR actual history.history values

train_acc = [
    0.6953, 0.7392, 0.7701, 0.7996, 0.8191,
    0.8331, 0.8416, 0.8499, 0.8582, 0.8641,
    0.8695, 0.8749, 0.8781, 0.8829, 0.8866
]

val_acc = [
    0.7588, 0.7709, 0.7920, 0.8174, 0.8239,
    0.8454, 0.8546, 0.8704, 0.8832, 0.8853,
    0.8859, 0.8845, 0.8991, 0.8980, 0.8938
]

train_loss = [
    0.5795, 0.5196, 0.4713, 0.4276, 0.3977,
    0.3738, 0.3558, 0.3405, 0.3247, 0.3135,
    0.3032, 0.2925, 0.2852, 0.2775, 0.2685
]

val_loss = [
    0.5055, 0.4756, 0.4391, 0.3993, 0.3749,
    0.3451, 0.3168, 0.2980, 0.2857, 0.2790,
    0.2697, 0.2678, 0.2471, 0.2487, 0.2518
]

epochs = range(1, len(train_acc) + 1)

# ==============================
# Training vs Validation Accuracy
# ==============================

plt.figure(figsize=(8,5))

plt.plot(epochs, train_acc, marker='o', label='Training Accuracy')
plt.plot(epochs, val_acc, marker='o', label='Validation Accuracy')

plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.savefig('training_vs_validation_accuracy.png')
plt.show()

# ==============================
# Training vs Validation Loss
# ==============================

plt.figure(figsize=(8,5))

plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='o', label='Validation Loss')

plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig('training_vs_validation_loss.png')
plt.show()