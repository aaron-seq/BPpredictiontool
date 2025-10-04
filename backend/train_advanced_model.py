"""
Advanced model training script with comprehensive evaluation and visualization.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import json

# Add backend directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import config
from data_processing.advanced_preprocessor import preprocessor
from models.advanced_lstm_model import bp_model
from utils.signal_processing import AdvancedSignalProcessor
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Advanced model trainer with comprehensive evaluation."""
    
    def __init__(self):
        self.model = bp_model
        self.preprocessor = preprocessor
        self.training_history = None
        
    def prepare_data(self):
        """Load and prepare training data."""
        logger.info("Loading and preprocessing data...")
        
        try:
            X_train, X_val, y_train, y_val = self.preprocessor.load_and_preprocess_data()
            
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Training target shape: {y_train.shape}")
            logger.info(f"Validation data shape: {X_val.shape}")
            logger.info(f"Validation target shape: {y_val.shape}")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """Train the advanced model."""
        logger.info("Starting model training...")
        
        # Build and compile model
        model = self.model.build_model(input_shape=X_train.shape[1:])
        self.model.compile_model(model)
        
        # Print model summary
        logger.info("Model Architecture:")
        model.summary(print_fn=logger.info)
        
        # Train model
        model_save_path = str(config.data.models_directory / 'advanced_bp_model.keras')
        
        history = self.model.train_model(
            X_train, y_train,
            X_val, y_val,
            model_save_path
        )
        
        self.training_history = history
        logger.info("Model training completed successfully!")
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation."""
        logger.info("Evaluating model performance...")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = self.model.evaluate_model(X_test, y_test)
        
        logger.info("Model Performance Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Additional analysis
        self._detailed_analysis(y_test, predictions)
        
        return metrics, predictions
    
    def _detailed_analysis(self, y_true, y_pred):
        """Perform detailed analysis of predictions."""
        logger.info("Performing detailed analysis...")
        
        # Convert to mmHg for analysis
        y_true_mmhg = y_true * 150
        y_pred_mmhg = y_pred * 150
        
        # Calculate per-sample metrics
        mae_per_sample = np.mean(np.abs(y_true_mmhg - y_pred_mmhg), axis=1)
        rmse_per_sample = np.sqrt(np.mean((y_true_mmhg - y_pred_mmhg) ** 2, axis=1))
        
        logger.info(f"Mean MAE per sample: {np.mean(mae_per_sample):.2f} mmHg")
        logger.info(f"Mean RMSE per sample: {np.mean(rmse_per_sample):.2f} mmHg")
        logger.info(f"Std MAE per sample: {np.std(mae_per_sample):.2f} mmHg")
        logger.info(f"Std RMSE per sample: {np.std(rmse_per_sample):.2f} mmHg")
        
        # Blood pressure range analysis
        self._bp_range_analysis(y_true_mmhg, y_pred_mmhg)
    
    def _bp_range_analysis(self, y_true, y_pred):
        """Analyze performance across different BP ranges."""
        logger.info("Analyzing performance across BP ranges...")
        
        # Define BP ranges
        ranges = {
            'Low (< 90)': y_true.max(axis=1) < 90,
            'Normal (90-120)': (y_true.max(axis=1) >= 90) & (y_true.max(axis=1) < 120),
            'Elevated (120-140)': (y_true.max(axis=1) >= 120) & (y_true.max(axis=1) < 140),
            'High (>= 140)': y_true.max(axis=1) >= 140
        }
        
        for range_name, mask in ranges.items():
            if np.any(mask):
                range_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                range_rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
                sample_count = np.sum(mask)
                
                logger.info(f"{range_name}: MAE={range_mae:.2f} mmHg, "
                           f"RMSE={range_rmse:.2f} mmHg, "
                           f"Samples={sample_count}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        if self.training_history is None:
            logger.warning("No training history available")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 3, 1)
        plt.plot(self.training_history['loss'], label='Training Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(2, 3, 2)
        plt.plot(self.training_history['mae'], label='Training MAE')
        plt.plot(self.training_history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        # Plot custom BP accuracy
        plt.subplot(2, 3, 3)
        plt.plot(self.training_history['_custom_bp_accuracy'], label='Training BP Accuracy')
        plt.plot(self.training_history['val__custom_bp_accuracy'], label='Validation BP Accuracy')
        plt.title('Blood Pressure Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
    
    def save_training_report(self, metrics, save_path=None):
        """Save comprehensive training report."""
        if save_path is None:
            save_path = config.data.logs_directory / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'sequence_length': config.model.sequence_length,
                'lstm_units': config.model.lstm_units,
                'batch_size': config.model.batch_size,
                'epochs': config.model.epochs,
                'learning_rate': config.model.learning_rate,
                'dropout_rate': config.model.dropout_rate
            },
            'training_metrics': metrics,
            'training_history': self.training_history
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {save_path}")

def main():
    """Main training function."""
    logger.info("Starting advanced model training...")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    try:
        # Prepare data
        X_train, X_val, y_train, y_val = trainer.prepare_data()
        
        # Train model
        history = trainer.train_model(X_train, X_val, y_train, y_val)
        
        # Evaluate model
        metrics, predictions = trainer.evaluate_model(X_val, y_val)
        
        # Save training plots
        plot_path = config.data.logs_directory / 'training_history.png'
        trainer.plot_training_history(str(plot_path))
        
        # Save training report
        trainer.save_training_report(metrics)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {config.data.models_directory / 'advanced_bp_model.keras'}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main()
