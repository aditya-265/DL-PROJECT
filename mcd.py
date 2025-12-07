import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- FIX FOR PYLANCE IMPORT ERROR ---
# Instead of importing 'preprocess_input' directly, we import the module alias.
# This prevents the "Import could not be resolved" red line in VS Code.
import tensorflow.keras.applications.efficientnet as efficientnet
from tensorflow.keras.applications import EfficientNetB0

# Create the shortcut variable so your code still works as expected
preprocess_input = efficientnet.preprocess_input 

from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MangoDiseaseClassifier:
    def __init__(self, dataset_path="./dataset"):
        self.dataset_path = dataset_path
        self.model = None
        self.history = None
        self.train_gen = None
        self.val_gen = None
        self.class_names = []
        
        # Configuration
        self.IMG_HEIGHT = 224
        self.IMG_WIDTH = 224
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.NUM_CLASSES = 8
        
        # Create models directory
        os.makedirs("./models", exist_ok=True)
    
    def explore_dataset(self):
        """Explore and visualize the dataset"""
        if not os.path.exists(self.dataset_path):
            print(f"Dataset directory '{self.dataset_path}' not found!")
            print("Please create the dataset folder with the following structure:")
            print("""
            dataset/
            ├── Anthracnose/
            ├── Bacterial Canker/
            ├── Cutting Weevil/
            ├── Die Back/
            ├── Gall Midge/
            ├── Powdery Mildew/
            ├── Sooty Mould/
            └── Healthy/
            """)
            return False
        
        # Get class information
        self.class_names = sorted([d for d in os.listdir(self.dataset_path) 
                                 if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        print("=== Dataset Exploration ===")
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Count images per class
        class_counts = {}
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(images)
            print(f"  {class_name}: {len(images)} images")
        
        # Plot class distribution
        self._plot_class_distribution(class_counts)
        return True
    
    def _plot_class_distribution(self, class_counts):
        """Plot class distribution"""
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        bars = plt.bar(class_counts.keys(), class_counts.values(), 
                      color='lightgreen', edgecolor='darkgreen')
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Disease Types')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, class_counts.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        plt.pie(class_counts.values(), labels=class_counts.keys(), 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Percentage Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        if not os.path.exists(self.dataset_path):
            print("Dataset path not found!")
            return False
        

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,   # <--- THIS IS THE KEY FIX
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Validation data generator
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,   # <--- THIS IS THE KEY FIX
            validation_split=0.2
        )
        
        # Training generator
        self.train_gen = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        self.val_gen = val_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_names = list(self.train_gen.class_indices.keys())
        
        print(f"Training samples: {self.train_gen.samples}")
        print(f"Validation samples: {self.val_gen.samples}")
        print(f"Classes: {self.class_names}")
        
        return True
    
    def create_model(self, model_type='efficientnet'):
        """Create the model architecture"""
        if model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)
            )
            base_model.trainable = False
            
            self.model = Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(len(self.class_names), activation='softmax')
            ])
            
        elif model_type == 'custom_cnn':
            self.model = Sequential([
                # First block
                Conv2D(32, (3, 3), activation='relu', padding='same', 
                      input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                # Second block
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                # Third block
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                # Classifier
                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(len(self.class_names), activation='softmax')
            ])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"{model_type.upper()} model created successfully!")
        return self.model
    
    def train(self, model_type='efficientnet'):
        """Train the model"""
        if self.train_gen is None:
            print("Please create data generators first!")
            return False
        
        # Create model
        self.create_model(model_type)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
            ModelCheckpoint(f'./models/best_{model_type}_model.h5', 
                          monitor='val_accuracy', save_best_only=True)
        ]
        
        print("Starting training...")
        self.history = self.model.fit(
            self.train_gen,
            epochs=self.EPOCHS,
            validation_data=self.val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        self.model.save(f'./models/final_{model_type}_model.h5')
        print("Training completed and model saved!")
        
        return True
    
    def evaluate(self):
        """Evaluate the model"""
        if self.model is None or self.val_gen is None:
            print("Model or validation data not available!")
            return False
        
        print("Evaluating model...")
        
        # Reset generator
        self.val_gen.reset()
        
        # Get predictions
        y_pred = self.model.predict(self.val_gen)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.val_gen.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return True
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_image(self, image_path):
        """Predict disease from a single image"""
        if self.model is None:
            print("No model loaded! Please train or load a model first.")
            return None
        
        if not os.path.exists(image_path):
            print(f"Image path '{image_path}' not found!")
            return None
        
        # Preprocess image
        img = Image.open(image_path)
        original_img = img.copy()
        img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
        
        # --- FIX STARTS HERE ---
        # 1. Do NOT divide by 255.0 here. EfficientNet expects raw 0-255 values.
        img_array = np.array(img)
        
        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. Use the specific preprocessing function for EfficientNet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_array = preprocess_input(img_array)
        # --- FIX ENDS HERE ---
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = self.class_names[predicted_idx]
        
        # Display results
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title(f'Input Image', fontsize=12)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        # Plot probabilities
        y_pos = np.arange(len(self.class_names))
        plt.barh(y_pos, predictions[0], color='skyblue')
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Probability')
        plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nPrediction Results:")
        print(f"  Disease: {predicted_class}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"\nAll probabilities:")
        for i, prob in enumerate(predictions[0]):
            print(f"  {self.class_names[i]}: {prob:.2%}")
        
        return predicted_class, confidence
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model path '{model_path}' not found!")
            return False

# Utility function for quick testing
def quick_test():
    """Quick test function"""
    classifier = MangoDiseaseClassifier()
    
    # Explore dataset
    if classifier.explore_dataset():
        # Create data generators
        if classifier.create_data_generators():
            print("\\nData generators created successfully!")
            
            # You can now train or load a model
            print("\\nTo train a model, run:")
            print("classifier.train(model_type='efficientnet')  # or 'custom_cnn'")
            print("\\nTo load a pre-trained model, run:")
            print("classifier.load_model('./models/your_model.h5')")
            print("\\nTo predict an image, run:")
            print("classifier.predict_image('path_to_your_image.jpg')")

if __name__ == "__main__":
    quick_test()