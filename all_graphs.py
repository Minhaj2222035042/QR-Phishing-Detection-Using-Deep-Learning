import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Model Accuracy Graph
# =========================

models = ["CNN", "MobileNetV2", "ResNet50", "EfficientNetB0", "DenseNet121", "VGG16", "InceptionV3"]
accuracy = [99, 99, 75, 99, 73, 72, 89]

plt.figure(figsize=(12,6))
bars = plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0,110)

for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 1, f"{acc}%", ha="center")

plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()


# =========================
# 2. Precision Comparison
# =========================

precision = [0.99, 0.99, 0.75, 0.99, 0.73, 0.79, 0.89]

plt.figure(figsize=(12,6))
bars = plt.bar(models, precision)
plt.title("Model Precision Comparison")
plt.xlabel("Models")
plt.ylabel("Precision")
plt.ylim(0,1.1)

for bar, value in zip(bars, precision):
    plt.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.2f}", ha="center")

plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("precision_comparison.png", dpi=300)
plt.show()


# =========================
# 3. Recall Comparison
# =========================

recall = [0.99, 0.99, 0.75, 0.99, 0.73, 0.72, 0.89]

plt.figure(figsize=(12,6))
bars = plt.bar(models, recall)
plt.title("Model Recall Comparison")
plt.xlabel("Models")
plt.ylabel("Recall")
plt.ylim(0,1.1)

for bar, value in zip(bars, recall):
    plt.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.2f}", ha="center")

plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("recall_comparison.png", dpi=300)
plt.show()


# =========================
# 4. F1-Score Comparison
# =========================

f1_score = [0.99, 0.99, 0.75, 0.99, 0.72, 0.69, 0.89]

plt.figure(figsize=(12,6))
bars = plt.bar(models, f1_score)
plt.title("Model F1-Score Comparison")
plt.xlabel("Models")
plt.ylabel("F1-Score")
plt.ylim(0,1.1)

for bar, value in zip(bars, f1_score):
    plt.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.2f}", ha="center")

plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("f1_score_comparison.png", dpi=300)
plt.show()


# =========================
# 5. Combined Metrics Graph
# =========================

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(14,7))

plt.bar(x - width*1.5, accuracy, width, label="Accuracy (%)")
plt.bar(x - width/2, [p*100 for p in precision], width, label="Precision (%)")
plt.bar(x + width/2, [r*100 for r in recall], width, label="Recall (%)")
plt.bar(x + width*1.5, [f*100 for f in f1_score], width, label="F1-Score (%)")

plt.title("Performance Comparison of Models")
plt.xlabel("Models")
plt.ylabel("Score (%)")
plt.xticks(x, models, rotation=25)
plt.ylim(0,110)
plt.legend()
plt.tight_layout()
plt.savefig("combined_model_metrics.png", dpi=300)
plt.show()


# =========================
# 6. Confusion Matrix Example for EfficientNetB0
# =========================

cm = np.array([
    [11526, 185],
    [4, 15684]
])

plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix - EfficientNetB0")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

labels = ["Legitimate", "Phishing"]
plt.xticks([0,1], labels)
plt.yticks([0,1], labels)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("efficientnet_confusion_matrix.png", dpi=300)
plt.show()