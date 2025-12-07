from mcd import MangoDiseaseClassifier
import os

def main():
    # New line - pointing specifically to your processed data folder
    classifier = MangoDiseaseClassifier(dataset_path="./DataSet/process data")
    
    print("=== Mango Leaf Disease Classifier ===\\n")
    
    # Step 1: Explore dataset
    print("1. Exploring dataset...")
    if not classifier.explore_dataset():
        return
    
    # Step 2: Create data generators
    print("\\n2. Creating data generators...")
    if not classifier.create_data_generators():
        return
    
    # Step 3: Ask user what they want to do
    print("\\n3. Choose an option:")
    print("   [1] Train a new model")
    print("   [2] Load existing model and predict")
    print("   [3] Just explore data (no training)")
    
    choice = input("\\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Train new model
        model_type = input("Choose model type (efficientnet/custom_cnn): ").strip().lower()
        if model_type not in ['efficientnet', 'custom_cnn']:
            model_type = 'efficientnet'
        
        print(f"\\nTraining {model_type} model...")
        classifier.train(model_type=model_type)
        
        # Evaluate model
        print("\\nEvaluating model...")
        classifier.evaluate()
        
        # Plot training history
        classifier.plot_training_history()
        
    elif choice == "2":
        # Load existing model
        model_path = input("Enter model path (or press enter for default): ").strip()
        if not model_path:
            model_path = "./models/best_efficientnet_model.h5"
        
        if classifier.load_model(model_path):
            # Predict on test image
            test_image = input("Enter test image path: ").strip()
            if test_image and os.path.exists(test_image):
                classifier.predict_image(test_image)
            else:
                print("Invalid image path!")
    
    elif choice == "3":
        print("\\nData exploration completed!")
        print("\\nYou can now:")
        print("  - Use classifier.train() to train a model")
        print("  - Use classifier.predict_image('image_path') to test predictions")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()