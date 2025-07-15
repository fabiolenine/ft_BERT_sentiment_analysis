# Examples Directory

This directory contains practical examples demonstrating how to use the BERT Sentiment Analysis project.

## 📁 Files Overview

### 1. `quick_start.py` - Quick Start Guide
A simple example to get you started quickly with the basic functionality.

**What it does:**
- Initializes the BERT trainer
- Loads and prepares a small dataset sample
- Trains the model with minimal configuration
- Evaluates performance
- Saves the model
- Demonstrates model versioning

**Usage:**
```bash
cd examples
python quick_start.py
```

**Expected output:**
- Training logs and progress
- Model evaluation metrics
- Saved model information

### 2. `api_usage.py` - API Usage Examples
Comprehensive examples for using the FastAPI application.

**What it does:**
- Tests API health endpoint
- Demonstrates sentiment prediction requests
- Shows error handling
- Benchmarks API performance
- Includes both sync and async examples

**Prerequisites:**
Start the API server first:
```bash
python api.py
```

**Usage:**
```bash
cd examples
python api_usage.py
```

**Features demonstrated:**
- Synchronous API calls
- Asynchronous batch processing
- Error handling
- Performance benchmarking
- Request/response format

### 3. `custom_training.py` - Advanced Training Examples
Advanced examples showing customization options.

**What it does:**
- **Experiment 1**: Hyperparameter tuning with different configurations
- **Experiment 2**: Model comparison (different BERT variants)
- **Experiment 3**: Advanced training techniques

**Usage:**
```bash
cd examples
python custom_training.py
```

**Experiments included:**
- Batch size optimization
- Learning rate tuning
- Sequence length comparison
- Model architecture comparison
- Advanced training techniques

## 🚀 Getting Started

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure proper directory structure:**
   ```
   ft_BERT_sentiment_analysis/
   ├── src/
   │   └── bert_sentiment_trainer.py
   ├── examples/
   │   ├── quick_start.py
   │   ├── api_usage.py
   │   └── custom_training.py
   └── api.py
   ```

### Running Examples

1. **Start with the quick start example:**
   ```bash
   python examples/quick_start.py
   ```

2. **For API examples, start the server first:**
   ```bash
   # Terminal 1: Start API server
   python api.py
   
   # Terminal 2: Run API examples
   python examples/api_usage.py
   ```

3. **For advanced training examples:**
   ```bash
   python examples/custom_training.py
   ```

## 📊 Expected Outputs

### Quick Start Example
```
🚀 BERT Sentiment Analysis - Quick Start Example
==================================================

1. Initializing BERT Sentiment Trainer...
2. Loading and preparing data...
Using 1000 samples for quick demo
3. Training the model...
4. Evaluating the model...
Test Results:
  Accuracy: 0.8550
  Precision: 0.8570
  Recall: 0.8550
  F1-Score: 0.8501
  AUC-ROC: 0.8876
5. Saving the model...
6. Checking for production promotion...
✅ Model promoted to production!
🎉 Quick start demo completed successfully!
```

### API Usage Example
```
🌐 Testing BERT Sentiment Analysis API (Synchronous)
============================================================

1. Testing Health Check...
Health Status: 200

2. Testing Sentiment Predictions...
📝 Test 1:
   Text: Eu adorei este produto! É excelente e recomendo...
   Sentiment: positivo
   Confidence: 0.9876
   Response time: 0.234s

📊 Concurrent Processing Results:
Total requests: 5
Total time: 0.456s
Average time per request: 0.091s
```

### Custom Training Example
```
🧪 Custom Training Experiments
===================================

🔬 Experiment 1: Hyperparameter Tuning
=============================================

🚀 Testing configuration: config_1_small_batch
✅ Configuration config_1_small_batch completed
   F1-Score: 0.8234
   Accuracy: 0.8200

🏆 Best configuration: config_2_large_batch
   Best F1-Score: 0.8456
```

## 🔧 Customization

### Modifying Examples

You can customize the examples by:

1. **Changing hyperparameters:**
   ```python
   # In quick_start.py
   trainer.train(
       batch_size=16,    # Increase batch size
       epochs=3,         # More epochs
       learning_rate=1e-5  # Different learning rate
   )
   ```

2. **Using different models:**
   ```python
   # In custom_training.py
   trainer = BERTSentimentTrainer(
       model_name='your-custom-model',
       max_length=512,
       num_labels=2
   )
   ```

3. **Adding new experiments:**
   ```python
   def experiment_4_custom():
       # Your custom experiment
       pass
   ```

### Adding New Examples

To add a new example:

1. Create a new Python file in the `examples/` directory
2. Import the necessary modules
3. Follow the existing code structure
4. Update this README with your new example

## 📝 Notes

- **GPU Usage**: Examples will automatically use GPU if available
- **Memory**: Large batch sizes may require more GPU memory
- **Time**: Full training examples may take several minutes
- **Logs**: All training logs are saved to `logs/train/`
- **Models**: Trained models are saved to `models/experiments/`

## 🐛 Troubleshooting

### Common Issues

1. **ImportError**: Make sure to run from the project root directory
2. **CUDA out of memory**: Reduce batch size or max_length
3. **API connection refused**: Start the API server first
4. **Model loading fails**: Check if the model files exist

### Debug Tips

- Check the logs in `logs/train/bert_sentiment_training.log`
- Use smaller sample sizes for faster testing
- Ensure all dependencies are installed
- Check GPU availability with `torch.cuda.is_available()`

## 🔗 Related Files

- **Main trainer**: `src/bert_sentiment_trainer.py`
- **API server**: `api.py`
- **Configuration**: `requirements.txt`
- **Documentation**: `README.md`

---

**Happy experimenting!** 🚀

For more information, check the main project README or submit an issue on GitHub.