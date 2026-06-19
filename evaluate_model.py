import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

# Test dataset
test_dir = "dataset_split/validation"

# Models list
models = [
    "qr_phishing_detector.h5",
    "mobilenet_qr_model.h5",
    "resnet_qr_model.h5",
    "efficientnet_model.h5",
    "densenet_model.h5",
    "vgg16_model.h5",
    "inception_model.h5"
]

for model_file in models:

    print("\n==============================")
    print("Evaluating:", model_file)
    print("==============================")

    if "efficientnet" in model_file:
        datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)

    elif "vgg16" in model_file:
        datagen = ImageDataGenerator(preprocessing_function=vgg16_preprocess)

    elif "resnet" in model_file:
        datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)

    elif "densenet" in model_file:
        datagen = ImageDataGenerator(preprocessing_function=densenet_preprocess)

    elif "inception" in model_file:
        datagen = ImageDataGenerator(preprocessing_function=inception_preprocess)

    else:
        datagen = ImageDataGenerator(rescale=1./255)

    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )

    model = tf.keras.models.load_model(model_file)

    predictions = model.predict(test_data)
    predictions = (predictions > 0.5)

    y_true = test_data.classes

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, predictions))

    print("\nClassification Report")
    print(classification_report(y_true, predictions, zero_division=0))