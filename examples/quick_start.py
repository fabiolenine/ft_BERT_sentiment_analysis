#!/usr/bin/env python3
"""
Quick Start Example for BERT Sentiment Analysis

This script demonstrates how to quickly get started with the BERT sentiment analysis project.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bert_sentiment_trainer import BERTSentimentTrainer, ModelVersionManager

def main():
    print("🚀 BERT Sentiment Analysis - Quick Start Example")
    print("=" * 50)
    
    # 1. Initialize the trainer
    print("\n1. Initializing BERT Sentiment Trainer...")
    trainer = BERTSentimentTrainer(
        model_name='neuralmind/bert-base-portuguese-cased',
        max_length=128,
        num_labels=2
    )
    
    # 2. Load and prepare data
    print("\n2. Loading and preparing data...")
    texts, labels = trainer.load_data()
    
    # Use a small subset for quick demo
    sample_size = 1000
    texts_sample = texts[:sample_size]
    labels_sample = labels[:sample_size]
    
    print(f"Using {sample_size} samples for quick demo")
    
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(
        texts_sample, 
        labels_sample, 
        seed_random_state=42
    )
    
    # 3. Train the model
    print("\n3. Training the model...")
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,  # Small batch size for quick demo
        epochs=1,      # Single epoch for quick demo
        learning_rate=2e-5
    )
    
    # 4. Evaluate the model
    print("\n4. Evaluating the model...")
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    _, _, test_metrics = trainer.evaluate(test_loader)
    
    print(f"Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    # 5. Save the model
    print("\n5. Saving the model...")
    version_manager = ModelVersionManager()
    
    hyperparameters = {
        'max_length': 128,
        'batch_size': 8,
        'epochs': 1,
        'learning_rate': 2e-5
    }
    
    experiment_path = version_manager.save_experiment(
        trainer, 
        test_metrics, 
        hyperparameters, 
        experiment_name="quick_start_demo"
    )
    
    print(f"Model saved to: {experiment_path}")
    
    # 6. Try to promote to production
    print("\n6. Checking for production promotion...")
    promoted = version_manager.promote_to_production(experiment_path)
    
    if promoted:
        print("✅ Model promoted to production!")
    else:
        print("ℹ️  Model not promoted (existing model is better)")
    
    print("\n🎉 Quick start demo completed successfully!")
    print("Check the logs directory for detailed training logs.")

if __name__ == "__main__":
    main()