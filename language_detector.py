"""
Language Detection Module

This module provides functionality to detect the language of individual words
using a character-level CNN model.

It can be used:
1. As an imported module in Python code
2. As a command-line script with arguments
3. As an interactive tool when run directly

Supported languages: Azerbaijani (az), English (en), Russian (ru)
"""

import torch
import pickle
import time
import traceback
import argparse
import sys
from typing import Tuple, Optional, Dict, Any, List, Union


class LanguageDetector:
    """Language detection model wrapper for word language identification"""
    
    def __init__(self, model_path: str = 'char_cnn_lang_classifier_optimized.pt', 
                 metadata_path: str = 'model_metadata.pkl') -> None:
        """
        Initialize the language detector
        
        Args:
            model_path: Path to the trained model
            metadata_path: Path to the model metadata
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load(self) -> bool:
        """
        Load the model and metadata
        
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load optimized model
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_supported_languages(self) -> List[str]:
        """
        Get the list of supported languages
        
        Returns:
            List[str]: List of language codes
        """
        if self.metadata is None:
            return []
        return list(self.metadata['label_encoder'].classes_)
    
    def predict(self, word: str) -> Tuple[str, float, float, List[str]]:
        """
        Predict the language of a word
        
        Args:
            word: The word to detect language for
            
        Returns:
            tuple: (language, confidence, inference_time, unknown_chars)
        """
        if self.model is None or self.metadata is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Check for unknown characters
        unknown_chars = [char for char in word if char not in self.metadata['char_to_idx']]
        
        # Get necessary data from metadata
        char_to_idx = self.metadata['char_to_idx']
        label_encoder = self.metadata['label_encoder']
        max_len = self.metadata['max_word_length']
        
        # Convert word to sequence of indices
        word_tensor = torch.zeros(1, max_len, dtype=torch.long)
        for i, char in enumerate(word[:max_len]):
            word_tensor[0, i] = char_to_idx.get(char, 0)
        
        # Move tensor to the appropriate device
        word_tensor = word_tensor.to(self.device)
        
        # Make prediction with time measurement
        start_time = time.time()
        with torch.no_grad():
            output = self.model(word_tensor)
        end_time = time.time()
        
        # Get results
        probs = torch.softmax(output, dim=1)
        confidence, idx = torch.max(probs, dim=1)
        language = label_encoder.inverse_transform([idx.item()])[0]
        inference_time = (end_time - start_time) * 1000  # ms
        
        return language, confidence.item(), inference_time, unknown_chars

    def batch_predict(self, words: List[str]) -> List[Dict[str, Union[str, float, List[str]]]]:
        """
        Predict languages for a batch of words
        
        Args:
            words: List of words to detect language for
            
        Returns:
            List of dictionaries with prediction results
        """
        results = []
        for word in words:
            try:
                language, confidence, inference_time, unknown_chars = self.predict(word)
                results.append({
                    'word': word,
                    'language': language,
                    'confidence': confidence,
                    'inference_time': inference_time,
                    'unknown_chars': unknown_chars
                })
            except Exception as e:
                results.append({
                    'word': word,
                    'error': str(e)
                })
        return results


def print_result(word: str, language: str, confidence: float, 
                inference_time: float, unknown_chars: List[str] = None) -> None:
    """
    Display detection result in a nice format
    
    Args:
        word: The analyzed word
        language: Detected language
        confidence: Confidence score (0-1)
        inference_time: Detection time in ms
        unknown_chars: List of characters not in the model vocabulary
    """
    print("\n" + "="*50)
    print(f"WORD: {word}")
    print("-"*50)
    print(f"LANGUAGE: {language}")
    print(f"CONFIDENCE: {confidence:.2%}")
    print(f"DETECTION TIME: {inference_time:.2f} ms")
    
    if unknown_chars and len(unknown_chars) > 0:
        print("-"*50)
        print(f"WARNING: {len(unknown_chars)} unknown character(s): {', '.join(unknown_chars)}")
    
    print("="*50)


def interactive_mode(detector: LanguageDetector) -> None:
    """
    Run the detector in interactive mode
    
    Args:
        detector: Initialized LanguageDetector instance
    """
    print("\nModel successfully loaded!")
    print("Supported languages:", ", ".join(detector.get_supported_languages()))
    print("\nTo exit, type 'q' or 'exit'")
    
    # Main program loop
    while True:
        print("\n" + "-"*50)
        word = input("Enter a word to detect its language: ").strip()
        
        # Check for exit command
        if word.lower() in ['q', 'exit', 'quit']:
            print("Exiting program...")
            break
        
        # Check for empty input
        if not word:
            print("Please enter a word.")
            continue
            
        # Determine the language of the word
        try:
            language, confidence, inference_time, unknown_chars = detector.predict(word)
            
            if unknown_chars:
                print(f"Warning: The following characters are not in the model vocabulary: {', '.join(unknown_chars)}")
                print("This may reduce the accuracy of language detection.")
            
            print_result(word, language, confidence, inference_time, unknown_chars)
        except Exception as e:
            print(f"Error determining language: {e}")
            # Add detailed error information for debugging
            print("\nError details:")
            traceback.print_exc()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Detect the language of a word or list of words")
    
    parser.add_argument("--word", "-w", type=str, help="Word to detect language for")
    parser.add_argument("--file", "-f", type=str, help="File with words (one per line)")
    parser.add_argument("--model", "-m", type=str, 
                        default="char_cnn_lang_classifier_optimized.pt",
                        help="Path to model file")
    parser.add_argument("--metadata", "-d", type=str, 
                        default="model_metadata.pkl",
                        help="Path to metadata file")
    parser.add_argument("--interactive", "-i", action="store_true", 
                        help="Run in interactive mode")
    parser.add_argument("--output", "-o", type=str, 
                        help="Output file for batch processing results (JSON format)")
    
    return parser.parse_args()


def main() -> None:
    """Main program function"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Print welcome message
    print("="*50)
    print("WORD LANGUAGE DETECTION")
    print("="*50)
    
    # Initialize detector
    detector = LanguageDetector(model_path=args.model, metadata_path=args.metadata)
    
    # Load model
    if not detector.load():
        print("Failed to load model or metadata. Make sure the files exist.")
        return 1
    
    # Handle different modes
    if args.interactive or (not args.word and not args.file):
        # Run in interactive mode
        interactive_mode(detector)
    elif args.word:
        # Process single word
        try:
            language, confidence, inference_time, unknown_chars = detector.predict(args.word)
            print_result(args.word, language, confidence, inference_time, unknown_chars)
        except Exception as e:
            print(f"Error processing word: {e}")
            return 1
    elif args.file:
        # Process file with words
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(words)} words from {args.file}...")
            results = detector.batch_predict(words)
            
            # Print summary
            success_count = sum(1 for r in results if 'error' not in r)
            print(f"\nProcessed {len(results)} words: {success_count} successful, {len(results) - success_count} failed")
            
            # Output detailed results if needed
            if args.output:
                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Detailed results saved to {args.output}")
            else:
                # Print first 5 results
                for i, result in enumerate(results[:5]):
                    if 'error' in result:
                        print(f"{i+1}. {result['word']}: ERROR - {result['error']}")
                    else:
                        print(f"{i+1}. {result['word']}: {result['language']} ({result['confidence']:.2%})")
                
                if len(results) > 5:
                    print(f"... and {len(results) - 5} more (use --output to save all results)")
        except Exception as e:
            print(f"Error processing file: {e}")
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())