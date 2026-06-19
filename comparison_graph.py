import matplotlib.pyplot as plt

models = ['CNN', 'MobileNetV2', 'ResNet50']
accuracy = [99.2, 99.28, 62]

plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("Models")
plt.savefig("model_accuracy_comparison.png", dpi=300)
plt.show()