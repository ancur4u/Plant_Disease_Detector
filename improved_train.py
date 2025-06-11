# improved_train.py - Better training parameters for plant disease detection

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ImprovedPlantDiseaseDetector:
    def __init__(self, img_height=224, img_width=224, num_classes=15):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        self.class_names = []
        
    def create_model(self):
        """Create an improved model with better architecture"""
        
        print("Creating improved model architecture...")
        
        # Use EfficientNetB0 as base
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Start with frozen base model
        base_model.trainable = False
        
        # Create improved model with better regularization
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        self.base_model = base_model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with better parameters"""
        
        print("Compiling model with improved settings...")
        
        # Use legacy optimizer for M1/M2/M3 compatibility
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model compiled successfully")
    
    def prepare_data(self, data_dir, validation_split=0.2, batch_size=16):
        """Prepare data with improved augmentation"""
        
        print("Preparing data with enhanced augmentation...")
        
        # Enhanced data augmentation for better training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Validation data with only rescaling
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training data with augmentation
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation data without augmentation
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        
        print(f"✅ Data prepared: {len(self.class_names)} classes")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        print(f"   Batch size: {batch_size}")
        
        return train_generator, val_generator
    
    def train_two_phase(self, train_gen, val_gen):
        """Two-phase training: frozen then unfrozen"""
        
        print("Starting two-phase training...")
        
        # Phase 1: Train with frozen base model
        print("\n📍 Phase 1: Training classifier head (frozen base)")
        
        callbacks_phase1 = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train phase 1
        history1 = self.model.fit(
            train_gen,
            epochs=25,
            validation_data=val_gen,
            callbacks=callbacks_phase1,
            verbose=1
        )
        
        # Phase 2: Unfreeze and fine-tune
        print("\n📍 Phase 2: Fine-tuning entire model (unfrozen base)")
        
        # Unfreeze the base model
        self.base_model.trainable = True
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=0.0001)
        
        callbacks_phase2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=6,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        # Train phase 2
        history2 = self.model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=callbacks_phase2,
            verbose=1
        )
        
        # Combine histories
        history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
        
        print("✅ Two-phase training completed!")
        return history
    
    def save_model(self, filepath='plant_disease_model.keras'):
        """Save the trained model in modern format"""
        
        print(f"Saving model to {filepath}...")
        
        # Save in modern Keras format
        self.model.save(filepath)
        
        # Also save in H5 format for compatibility
        self.model.save('plant_disease_model.h5')
        
        # Save class names
        with open('class_names.txt', 'w') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        
        print("✅ Model and class names saved!")
        print(f"   - Modern format: {filepath}")
        print(f"   - Legacy format: plant_disease_model.h5")
        print(f"   - Class names: class_names.txt")
    
    def evaluate_model(self, val_gen):
        """Evaluate the final model performance"""
        
        print("Evaluating final model performance...")
        
        # Evaluate on validation data
        val_loss, val_accuracy = self.model.evaluate(val_gen, verbose=0)
        
        print(f"📊 Final Model Performance:")
        print(f"   Validation Loss: {val_loss:.4f}")
        print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
        
        return val_loss, val_accuracy
    
    def plot_training_history(self, history):
        """Plot comprehensive training history"""
        
        print("Creating comprehensive training plots...")
        
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss
        plt.subplot(1, 3, 2)
        plt.plot(history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy improvement
        plt.subplot(1, 3, 3)
        final_acc = history['val_accuracy'][-1]
        epochs = range(1, len(history['accuracy']) + 1)
        plt.plot(epochs, history['val_accuracy'], 'o-', linewidth=2, markersize=4)
        plt.axhline(y=final_acc, color='r', linestyle='--', alpha=0.7)
        plt.title(f'Validation Accuracy Progress\nFinal: {final_acc:.1%}', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training plots saved as 'training_history.png'!")

def main():
    print("🌿 Improved Plant Disease Model Training")
    print("=" * 50)
    
    # Check dataset
    dataset_path = "plant_dataset/PlantVillage"
    if not Path(dataset_path).exists():
        print("❌ Dataset not found. Please run dataset setup first.")
        return False
    
    # Initialize improved detector
    detector = ImprovedPlantDiseaseDetector(num_classes=15)
    
    # Create and compile model
    model = detector.create_model()
    detector.compile_model()
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Prepare data with smaller batch size for better learning
    train_gen, val_gen = detector.prepare_data(dataset_path, batch_size=16)
    
    # Two-phase training
    try:
        history = detector.train_two_phase(train_gen, val_gen)
        
        # Evaluate final performance
        val_loss, val_accuracy = detector.evaluate_model(val_gen)
        
        # Save model
        detector.save_model()
        
        # Plot results
        detector.plot_training_history(history)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"✅ Final Accuracy: {val_accuracy:.1%}")
        print(f"✅ Model saved and ready to use")
        print(f"🚀 Run: streamlit run streamlit_app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()