# Language Detection Model

A character-level CNN model for detecting the language of a word. This model can accurately distinguish between Azerbaijani (az), English (en), and Russian (ru) words with high precision.

## Performance Metrics

The model achieves excellent performance across all supported languages:

```
              precision    recall  f1-score   support

          az       0.98      0.96      0.97     44773
          en       0.97      0.98      0.97     46163
          ru       1.00      1.00      1.00    152891

    accuracy                           0.99    243827
   macro avg       0.98      0.98      0.98    243827
weighted avg       0.99      0.99      0.99    243827
```

- **Overall Accuracy**: 98.99%
- **Average Inference Time**: 0.0421 ms per word

## Installation

```bash
# Clone the repository
git clone https://github.com/vrashad/language_detect_cnn.git
cd language_detect_cnn

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
torch
numpy
pandas
scikit-learn
```

## Usage

The language detector can be used in three different ways:

### 1. As a Python Module

```python
from language_detector import LanguageDetector

# Initialize the detector
detector = LanguageDetector()

# Load the model
if detector.load():
    # Process a single word
    language, confidence, inference_time, unknown_chars = detector.predict("hello")
    print(f"Word 'hello' is in {language} with {confidence:.2%} confidence")
    
    # Process multiple words
    words = ["hello", "привет", "salam"]
    results = detector.batch_predict(words)
    for result in results:
        print(f"{result['word']}: {result['language']} ({result['confidence']:.2%})")
```

### 2. From Command Line with Arguments

```bash
# Process a single word
python language_detector.py --word "hello"

# Process words from a file (one word per line)
python language_detector.py --file words.txt

# Save batch processing results to a JSON file
python language_detector.py --file words.txt --output results.json

# Specify custom model and metadata paths
python language_detector.py --word "hello" --model "my_model.pt" --metadata "my_metadata.pkl"
```

### 3. Interactive Mode

```bash
# Run in interactive mode
python language_detector.py --interactive

# Or simply run without arguments
python language_detector.py
```

## Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--word` | `-w` | Single word to detect language for |
| `--file` | `-f` | File with words (one per line) |
| `--model` | `-m` | Custom path to model file |
| `--metadata` | `-d` | Custom path to metadata file |
| `--interactive` | `-i` | Run in interactive mode |
| `--output` | `-o` | Output file for batch processing results (JSON format) |

## Model Architecture

The model uses a character-level Convolutional Neural Network (CNN) architecture:

- Character embedding layer
- Parallel convolutional layers with different kernel sizes (2 and 3) to capture different n-gram patterns
- Global max pooling
- Fully connected layer for classification

This architecture is particularly effective for language identification as it learns character-level patterns that are specific to each language.

## Technical Details

- **Framework**: PyTorch
- **Input**: Words converted to character indices (maximum length: 20 characters)
- **Output**: Language prediction (az, en, or ru)
- **Model Size**: Lightweight (optimized for inference)
- **Optimization**: TorchScript for faster inference

## Repository Contents

- `language_detector.py`: Main module with detector class and CLI functionality
- `char_cnn_lang_classifier.pt`: The trained PyTorch model
- `char_cnn_lang_classifier_optimized.pt`: TorchScript optimized model for inference
- `model_metadata.pkl`: Model metadata including character mappings and label encoders
- `requirements.txt`: Required Python packages


## Example Output

```
==================================================
WORD: hello
--------------------------------------------------
LANGUAGE: en
CONFIDENCE: 99.87%
DETECTION TIME: 0.42 ms
==================================================
```

## Batch Processing

When processing multiple words using the `--file` option, the tool can generate a detailed JSON report with the following information for each word:

```json
[
  {
    "word": "hello",
    "language": "en",
    "confidence": 0.9987,
    "inference_time": 0.42,
    "unknown_chars": []
  },
  {
    "word": "привет",
    "language": "ru",
    "confidence": 0.9998,
    "inference_time": 0.38,
    "unknown_chars": []
  }
]
```

## License

This project is licensed under the Apache 2.0 License

## Acknowledgments

- This model architecture was inspired by character-level CNN approaches to text classification
- Special thanks to all contributors who helped with dataset collection and cleaning

## Future Work

- Add support for more languages
- Experiment with LSTM and Transformer-based architectures
- Create a web API for language detection
- Optimize for mobile deployment
