import matplotlib.pyplot as plt

models = [
    "CNN",
    "MobileNetV2",
    "ResNet50",
    "EfficientNetB0",
    "DenseNet121",
    "VGG16",
    "InceptionV3"
]

accuracy = [99, 99, 75, 99, 73, 72, 89.9]

# Different colors
colors = [
    "green",
    "green",
    "orange",
    "green",
    "red",
    "red",
    "blue"
]

plt.figure(figsize=(12,6))

bars = plt.bar(models, accuracy, color=colors)

plt.title("Accuracy Comparison of Deep Learning Models", fontsize=16)
plt.xlabel("Models", fontsize=13)
plt.ylabel("Accuracy (%)", fontsize=13)

plt.ylim(0, 110)

# Grid
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Accuracy labels
for bar, acc in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        acc + 1,
        f"{acc}%",
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("final_model_comparison.png", dpi=300)

plt.show()