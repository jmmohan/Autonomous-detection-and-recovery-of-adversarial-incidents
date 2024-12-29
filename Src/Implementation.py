import tensorflow as tf
import numpy as np
import shap
from lime import lime_image
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier

# Step 1: Load and Prepare CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images to [0, 1]
y_train, y_test = y_train.flatten(), y_test.flatten()

# Step 2: Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Step 4: Create an ART Classifier for Adversarial Attacks
classifier = KerasClassifier(model=model)

# Step 5: Generate Adversarial Samples using FGSM
attack = FastGradientMethod(classifier)
x_test_adv = attack.generate(x=x_test)  # Adversarial examples for testing

# Step 6: Detect and Explain Adversarial Samples using SHAP and LIME

# 6.1: SHAP Explanation
explainer = shap.KernelExplainer(model.predict, x_train[:100])  # Using a sample for background
shap_values = explainer.shap_values(x_test_adv[:5])  # Explain top 5 adversarial samples
shap.summary_plot(shap_values[0], x_test_adv[:5])  # Plot SHAP explanations

# 6.2: LIME Explanation
def predict_fn(images):
    return model.predict(images)

explainer_lime = lime_image.LimeImageExplainer()
explanation = explainer_lime.explain_instance(x_test_adv[0], predict_fn, top_labels=5, hide_color=0, num_samples=1000)
explanation.show_in_browser()  # Visualize LIME explanation

# Step 7: Recovery Mechanism - Retrain Model on Clean Data
def retrain_model(model, x_train, y_train):
    print("Retraining model to recover from attack...")
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    return model

# Simulate Recovery after Adversarial Attack
model = retrain_model(model, x_train, y_train)

# Step 8: Continuous Learning - Simulate Feedback and Retraining
def continuous_learning(model, x_train, y_train, x_test, y_test):
    print("Performing continuous learning...")
    model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
    return model

# Simulate Continuous Learning
model = continuous_learning(model, x_train, y_train, x_test, y_test)

# Final Evaluation after Recovery and Continuous Learning
final_loss, final_accuracy = model.evaluate(x_test, y_test)
print(f"Final Model Accuracy: {final_accuracy:.4f}")
