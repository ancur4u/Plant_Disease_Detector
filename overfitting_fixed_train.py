# overfitting_fixed_train.py - Training script that prevents overfitting

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class OverfittingFixedDetector:
    def __init__(self, img_height=224, img_width=224, num_classes=15):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        self.class_names = []
        
    def create_regularized_model(self):
        """Create model with strong regularization to prevent overfitting"""
        
        print("Creating heavily regularized model...")
        
        # Use smaller, more controlled base model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Keep base model frozen throughout training
        base_model.trainable = False
        
        # Create model with heavy regularization
        model = tf.keras.Sequential([
            # Data augmentation layer (built into model)
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            
            # Base model
            base_model,
            
            # Heavily regularized classifier
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.7),  # High dropout
            tf.keras.layers.Dense(128, activation='relu', 
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.8),  # Very high dropout
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.0001):
        """Compile with conservative settings"""
        
        print("Compiling with anti-overfitting settings...")
        
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model compiled with regularization")
    
    def prepare_balanced_data(self, data_dir, validation_split=0.3, batch_size=32):
        """Prepare data with stronger validation split and less aggressive augmentation"""
        
        print("Preparing data with conservative augmentation...")
        
        # Less aggressive augmentation to prevent overfitting
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,  # Reduced from 30
            width_shift_range=0.1,  # Reduced from 0.2
            height_shift_range=0.1,  # Reduced from 0.2
            shear_range=0.1,  # Reduced from 0.2
            zoom_range=0.1,  # Reduced from 0.2
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Validation data with only rescaling
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training data
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation data
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        
        print(f"✅ Data prepared with larger validation set:")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        print(f"   Validation split: {validation_split*100}%")
        
        return train_generator, val_generator
    
    def train_conservative(self, train_gen, val_gen):
        """Conservative training focused on generalization"""
        
        print("Starting conservative training (single phase, frozen base)...")
        
        # Very conservative callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Focus on validation performance
                patience=10,  # More patience
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-8,
                verbose=1
            ),
            # Add model checkpoint to save best model
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("Training with focus on validation performance...")
        
        # Single phase training with frozen base
        history = self.model.fit(
            train_gen,
            epochs=50,  # More epochs but will early stop
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Conservative training completed!")
        return history
    
    def evaluate_generalization(self, val_gen):
        """Evaluate how well model generalizes"""
        
        print("Evaluating model generalization...")
        
        # Evaluate on validation data
        val_loss, val_accuracy = self.model.evaluate(val_gen, verbose=0)
        
        # Calculate overfitting indicators
        train_loss = self.model.history.history['loss'][-1] if hasattr(self.model, 'history') else 0
        train_accuracy = self.model.history.history['accuracy'][-1] if hasattr(self.model, 'history') else 0
        
        print(f"📊 Final Model Performance:")
        print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
        print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
        print(f"   Generalization Gap: {abs(train_accuracy - val_accuracy):.4f}")
        
        if abs(train_accuracy - val_accuracy) < 0.1:
            print("✅ Good generalization - low overfitting")
        elif abs(train_accuracy - val_accuracy) < 0.2:
            print("⚠️ Moderate generalization - some overfitting")
        else:
            print("❌ Poor generalization - significant overfitting")
        
        return val_loss, val_accuracy
    
    def save_model(self, filepath='plant_disease_model_v2.h5'):
        """Save the trained model"""
        
        print(f"Saving regularized model to {filepath}...")
        
        # Save in H5 format for compatibility
        self.model.save(filepath)
        self.model.save('plant_disease_model.h5')  # Override previous
        
        # Save class names
        with open('class_names.txt', 'w') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        
        print("✅ Model saved with anti-overfitting architecture!")
    
    def plot_training_analysis(self, history):
        """Plot training with overfitting analysis"""
        
        print("Creating overfitting analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0,0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0,0].set_title('Model Accuracy (Overfitting Check)', fontweight='bold')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0,1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0,1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0,1].set_title('Model Loss (Overfitting Check)', fontweight='bold')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Overfitting gap analysis
        if len(history.history['accuracy']) > 1:
            accuracy_gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
            axes[1,0].plot(accuracy_gap, 'r-', linewidth=2, label='Accuracy Gap')
            axes[1,0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Acceptable Gap')
            axes[1,0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
            axes[1,0].set_title('Overfitting Analysis', fontweight='bold')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Training - Validation Accuracy')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Final performance summary
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        axes[1,1].bar(['Training', 'Validation'], [final_train_acc, final_val_acc], 
                     color=['blue', 'orange'], alpha=0.7)
        axes[1,1].set_title(f'Final Performance\nGap: {abs(final_train_acc - final_val_acc):.3f}', fontweight='bold')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_ylim([0, 1])
        
        # Add performance annotations
        for i, v in enumerate([final_train_acc, final_val_acc]):
            axes[1,1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Overfitting analysis saved as 'overfitting_analysis.png'!")

def main():
    print("🌿 Anti-Overfitting Plant Disease Model Training")
    print("=" * 55)
    
    # Check dataset
    dataset_path = "plant_dataset/PlantVillage"
    if not Path(dataset_path).exists():
        print("❌ Dataset not found. Please run dataset setup first.")
        return False
    
    # Initialize detector with regularization
    detector = OverfittingFixedDetector(num_classes=15)
    
    # Create heavily regularized model
    model = detector.create_regularized_model()
    detector.compile_model()
    
    # Build the model to count parameters
    model.build((None, detector.img_height, detector.img_width, 3))
    print(f"Model parameters: {model.count_params():,}")
    print("🔒 Base model frozen to prevent overfitting")
    
    # Prepare data with larger validation split
    train_gen, val_gen = detector.prepare_balanced_data(dataset_path, validation_split=0.3, batch_size=32)
    
    # Conservative training
    try:
        history = detector.train_conservative(train_gen, val_gen)
        
        # Evaluate generalization
        val_loss, val_accuracy = detector.evaluate_generalization(val_gen)
        
        # Save model
        detector.save_model()
        
        # Plot analysis
        detector.plot_training_analysis(history)
        
        print(f"\n🎉 Anti-overfitting training completed!")
        print(f"✅ Final Validation Accuracy: {val_accuracy:.1%}")
        print(f"✅ Model focuses on generalization over memorization")
        print(f"🚀 Test with: streamlit run streamlit_app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()