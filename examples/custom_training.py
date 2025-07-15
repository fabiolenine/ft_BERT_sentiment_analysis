#!/usr/bin/env python3
"""
Custom Training Example for BERT Sentiment Analysis

This script demonstrates how to customize the training process with different
hyperparameters, datasets, and model configurations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bert_sentiment_trainer import BERTSentimentTrainer, ModelVersionManager
from torch.utils.data import DataLoader
import json

def experiment_1_hyperparameter_tuning():
    """Experiment 1: Hyperparameter tuning with different configurations."""
    print("🔬 Experiment 1: Hyperparameter Tuning")
    print("=" * 45)
    
    # Different hyperparameter configurations to test
    configs = [
        {
            "name": "config_1_small_batch",
            "max_length": 256,
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 2e-5
        },
        {
            "name": "config_2_large_batch",
            "max_length": 256,
            "batch_size": 16,
            "epochs": 2,
            "learning_rate": 1e-5
        },
        {
            "name": "config_3_longer_sequence",
            "max_length": 512,
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 3e-5
        }
    ]
    
    version_manager = ModelVersionManager()
    results = []
    
    for config in configs:
        print(f"\\n🚀 Testing configuration: {config['name']}")
        print(f"Configuration: {config}")
        
        # Initialize trainer with specific configuration
        trainer = BERTSentimentTrainer(
            model_name='neuralmind/bert-base-portuguese-cased',
            max_length=config['max_length'],
            num_labels=2
        )
        
        # Load and prepare data
        texts, labels = trainer.load_data()
        
        # Use subset for faster experimentation
        sample_size = 2000
        texts_sample = texts[:sample_size]
        labels_sample = labels[:sample_size]
        
        train_dataset, val_dataset, test_dataset = trainer.prepare_data(
            texts_sample, 
            labels_sample, 
            seed_random_state=42
        )
        
        # Train with specific configuration
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            learning_rate=config['learning_rate']
        )
        
        # Evaluate
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        _, _, test_metrics = trainer.evaluate(test_loader)
        
        # Save experiment
        experiment_path = version_manager.save_experiment(
            trainer, 
            test_metrics, 
            config, 
            experiment_name=config['name']
        )
        
        # Store results
        result = {
            'config': config,
            'metrics': test_metrics,
            'experiment_path': experiment_path
        }
        results.append(result)
        
        print(f"✅ Configuration {config['name']} completed")
        print(f"   F1-Score: {test_metrics['f1']:.4f}")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Compare results
    print("\\n📊 Experiment 1 Results Summary:")
    print("=" * 40)
    
    best_config = max(results, key=lambda x: x['metrics']['f1'])
    
    for result in results:
        config = result['config']
        metrics = result['metrics']
        
        print(f"\\n{config['name']}:")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Max Length: {config['max_length']}")
    
    print(f"\\n🏆 Best configuration: {best_config['config']['name']}")
    print(f"   Best F1-Score: {best_config['metrics']['f1']:.4f}")
    
    return best_config

def experiment_2_model_comparison():
    """Experiment 2: Compare different BERT model variants."""
    print("\\n🔬 Experiment 2: Model Comparison")
    print("=" * 35)
    
    # Different Portuguese BERT models to compare
    models = [
        {
            "name": "bert-base-portuguese-cased",
            "model_name": "neuralmind/bert-base-portuguese-cased",
            "description": "Standard Portuguese BERT"
        },
        {
            "name": "bert-base-portuguese-uncased",
            "model_name": "neuralmind/bert-base-portuguese-uncased",
            "description": "Uncased Portuguese BERT"
        }
    ]
    
    # Standard configuration
    config = {
        "max_length": 256,
        "batch_size": 8,
        "epochs": 1,
        "learning_rate": 2e-5
    }
    
    version_manager = ModelVersionManager()
    model_results = []
    
    for model_info in models:
        print(f"\\n🚀 Testing model: {model_info['name']}")
        print(f"Description: {model_info['description']}")
        
        try:
            # Initialize trainer with specific model
            trainer = BERTSentimentTrainer(
                model_name=model_info['model_name'],
                max_length=config['max_length'],
                num_labels=2
            )
            
            # Load and prepare data
            texts, labels = trainer.load_data()
            
            # Use subset for faster comparison
            sample_size = 1500
            texts_sample = texts[:sample_size]
            labels_sample = labels[:sample_size]
            
            train_dataset, val_dataset, test_dataset = trainer.prepare_data(
                texts_sample, 
                labels_sample, 
                seed_random_state=42
            )
            
            # Train
            trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                learning_rate=config['learning_rate']
            )
            
            # Evaluate
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
            _, _, test_metrics = trainer.evaluate(test_loader)
            
            # Save experiment
            experiment_name = f"model_comparison_{model_info['name']}"
            experiment_path = version_manager.save_experiment(
                trainer, 
                test_metrics, 
                {**config, "model_name": model_info['model_name']}, 
                experiment_name=experiment_name
            )
            
            # Store results
            result = {
                'model_info': model_info,
                'metrics': test_metrics,
                'experiment_path': experiment_path
            }
            model_results.append(result)
            
            print(f"✅ Model {model_info['name']} completed")
            print(f"   F1-Score: {test_metrics['f1']:.4f}")
            print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ Error with model {model_info['name']}: {e}")
    
    # Compare model results
    print("\\n📊 Experiment 2 Results Summary:")
    print("=" * 40)
    
    if model_results:
        best_model = max(model_results, key=lambda x: x['metrics']['f1'])
        
        for result in model_results:
            model_info = result['model_info']
            metrics = result['metrics']
            
            print(f"\\n{model_info['name']}:")
            print(f"  Description: {model_info['description']}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print(f"\\n🏆 Best model: {best_model['model_info']['name']}")
        print(f"   Best F1-Score: {best_model['metrics']['f1']:.4f}")
        
        return best_model
    else:
        print("❌ No successful model results to compare")
        return None

def experiment_3_advanced_training():
    """Experiment 3: Advanced training techniques."""
    print("\\n🔬 Experiment 3: Advanced Training Techniques")
    print("=" * 50)
    
    # Advanced configuration with different techniques
    config = {
        "max_length": 256,
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 2e-5,
        "techniques": ["warmup_scheduler", "gradient_clipping", "early_stopping"]
    }
    
    print(f"Configuration: {config}")
    
    # Initialize trainer
    trainer = BERTSentimentTrainer(
        model_name='neuralmind/bert-base-portuguese-cased',
        max_length=config['max_length'],
        num_labels=2
    )
    
    # Load and prepare data
    texts, labels = trainer.load_data()
    
    # Use larger sample for advanced training
    sample_size = 5000
    texts_sample = texts[:sample_size]
    labels_sample = labels[:sample_size]
    
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(
        texts_sample, 
        labels_sample, 
        seed_random_state=42
    )
    
    # Train with advanced techniques (already implemented in trainer)
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate']
    )
    
    # Evaluate
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    _, _, test_metrics = trainer.evaluate(test_loader)
    
    # Save experiment
    version_manager = ModelVersionManager()
    experiment_path = version_manager.save_experiment(
        trainer, 
        test_metrics, 
        config, 
        experiment_name="advanced_training"
    )
    
    print(f"\\n✅ Advanced training completed")
    print(f"   F1-Score: {test_metrics['f1']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    # Try to promote to production
    promoted = version_manager.promote_to_production(experiment_path)
    if promoted:
        print("\\n🎉 Model promoted to production!")
    else:
        print("\\nℹ️  Model not promoted (existing model is better)")
    
    return {
        'config': config,
        'metrics': test_metrics,
        'experiment_path': experiment_path,
        'promoted': promoted
    }

def main():
    """Run all custom training experiments."""
    print("🧪 Custom Training Experiments")
    print("=" * 35)
    print("This script will run multiple experiments to demonstrate")
    print("different training configurations and techniques.")
    print()
    
    # Run experiments
    experiment_1_result = experiment_1_hyperparameter_tuning()
    experiment_2_result = experiment_2_model_comparison()
    experiment_3_result = experiment_3_advanced_training()
    
    # Final summary
    print("\\n🎯 Final Summary")
    print("=" * 20)
    
    print("\\n📈 Best Results:")
    print(f"Hyperparameter Tuning: F1-Score {experiment_1_result['metrics']['f1']:.4f}")
    
    if experiment_2_result:
        print(f"Model Comparison: F1-Score {experiment_2_result['metrics']['f1']:.4f}")
    
    print(f"Advanced Training: F1-Score {experiment_3_result['metrics']['f1']:.4f}")
    
    print("\\n✅ All experiments completed!")
    print("Check the models/experiments/ directory for saved models.")
    print("Check the logs/train/ directory for detailed training logs.")

if __name__ == "__main__":
    main()