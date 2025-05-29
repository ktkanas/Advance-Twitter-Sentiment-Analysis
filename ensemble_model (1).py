"""
Advanced Ensemble Model for Sentiment Analysis
Author: Your Name
Description: Production-ready ensemble model combining traditional ML,
            deep learning, and transformer models for optimal performance.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional,
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, Attention
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Transformers
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, TrainingArguments, Trainer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. BERT models will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training and ensemble."""
    # General
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Traditional ML
    use_tfidf: bool = True
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 2)
    
    # Deep Learning
    max_sequence_length: int = 128
    embedding_dim: int = 100
    lstm_units: int = 64
    dropout_rate: float = 0.5
    epochs: int = 10
    batch_size: int = 32
    
    # Transformer
    transformer_model: str = "distilbert-base-uncased"
    max_length: int = 128
    
    # Ensemble
    ensemble_method: str = "weighted"  # "simple", "weighted", "stacking"
    

class TraditionalMLModels:
    """Traditional machine learning models for sentiment analysis."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.vectorizers = {}
        self.is_trained = False
        
    def create_models(self):
        """Initialize traditional ML models."""
        self.models = {
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=self.config.max_features,
                    ngram_range=self.config.ngram_range,
                    stop_words='english'
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ]),
            
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=self.config.max_features,
                    ngram_range=self.config.ngram_range,
                    stop_words='english'
                )),
                ('classifier', LogisticRegression(
                    random_state=self.config.random_state,
                    max_iter=1000
                ))
            ]),
            
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=self.config.max_features,
                    ngram_range=self.config.ngram_range,
                    stop_words='english'
                )),
                ('classifier', SVC(
                    kernel='linear',
                    probability=True,
                    random_state=self.config.random_state
                ))
            ]),
            
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=self.config.max_features,
                    ngram_range=self.config.ngram_range,
                    stop_words='english'
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state,
                    n_jobs=-1
                ))
            ])
        }
        
        logger.info(f"Created {len(self.models)} traditional ML models")
    
    def train(self, X_train: List[str], y_train: List[int]) -> Dict[str, float]:
        """Train all traditional ML models."""
        if not self.models:
            self.create_models()
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cv_folds,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Train on full dataset
            model.fit(X_train, y_train)
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            logger.info(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.is_trained = True
        return results
    
    def predict(self, X_test: List[str]) -> Dict[str, np.ndarray]:
        """Get predictions from all models."""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X_test)
        
        return predictions


class LSTMModel:
    """LSTM model for sentiment analysis."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        
    def create_model(self, vocab_size: int) -> tf.keras.Model:
        """Create LSTM model architecture."""
        model = Sequential([
            Embedding(
                vocab_size, 
                self.config.embedding_dim,
                input_length=self.config.max_sequence_length
            ),
            Bidirectional(LSTM(
                self.config.lstm_units,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate,
                return_sequences=True
            )),
            Bidirectional(LSTM(
                self.config.lstm_units // 2,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            )),
            Dense(64, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(32, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(3, activation='softmax')  # negative, neutral, positive
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, texts: List[str]) -> np.ndarray:
        """Tokenize and pad sequences."""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.config.max_features)
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.config.max_sequence_length)
    
    def train(self, X_train: List[str], y_train: List[int]) -> Dict:
        """Train LSTM model."""
        logger.info("Training LSTM model...")
        
        # Prepare data
        X_train_seq = self.prepare_data(X_train)
        vocab_size = len(self.tokenizer.word_index) + 1
        
        # Create model
        self.model = self.create_model(vocab_size)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-7)
        ]
        
        # Train model
        history = self.model.fit(
            X_train_seq, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        return {
            'history': history.history,
            'final_accuracy': max(history.history['val_accuracy'])
        }
    
    def predict(self, X_test: List[str]) -> np.ndarray:
        """Get predictions from LSTM model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_test_seq = self.prepare_data(X_test)
        return self.model.predict(X_test_seq)


class TransformerModel:
    """Transformer-based model for sentiment analysis."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_trained = False
        
    def load_pretrained(self):
        """Load pre-trained transformer model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.is_trained = True
            logger.info("Loaded pre-trained RoBERTa model")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            # Fallback to simpler model
            self.pipeline = pipeline("sentiment-analysis")
            self.is_trained = True
    
    def predict(self, X_test: List[str]) -> np.ndarray:
        """Get predictions from transformer model."""
        if not self.is_trained:
            self.load_pretrained()
        
        predictions = []
        batch_size = 32
        
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size]
            batch_predictions = self.pipeline(batch)
            
            # Convert to probability format
            batch_probs = []
            for pred in batch_predictions:
                if pred['label'] == 'NEGATIVE':
                    probs = [pred['score'], 1 - pred['score'], 0]
                elif pred['label'] == 'POSITIVE':
                    probs = [0, 1 - pred['score'], pred['score']]
                else:  # NEUTRAL
                    probs = [0, pred['score'], 1 - pred['score']]
                batch_probs.append(probs)
            
            predictions.extend(batch_probs)
        
        return np.array(predictions)


class SentimentEnsemble:
    """Ensemble model combining multiple approaches."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.traditional_models = TraditionalMLModels(self.config)
        self.lstm_model = LSTMModel(self.config)
        self.transformer_model = TransformerModel(self.config)
        
        self.label_encoder = LabelEncoder()
        self.weights = {}
        self.is_trained = False
        
    def train(self, texts: List[str], labels: List[str]) -> Dict:
        """Train all models in the ensemble."""
        logger.info("Starting ensemble training...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, y_encoded,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_encoded
        )
        
        results = {}
        
        # Train traditional models
        logger.info("Training traditional ML models...")
        traditional_results = self.traditional_models.train(X_train, y_train)
        results['traditional'] = traditional_results
        
        # Train LSTM model
        logger.info("Training LSTM model...")
        try:
            lstm_results = self.lstm_model.train(X_train, y_train)
            results['lstm'] = lstm_results
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            results['lstm'] = None
        
        # Load transformer model
        logger.info("Loading transformer model...")
        try:
            self.transformer_model.load_pretrained()
            results['transformer'] = {"status": "loaded"}
        except Exception as e:
            logger.error(f"Transformer loading failed: {e}")
            results['transformer'] = None
        
        # Calculate ensemble weights based on validation performance
        self._calculate_weights(X_val, y_val)
        
        self.is_trained = True
        logger.info("Ensemble training completed!")
        
        return results
    
    def _calculate_weights(self, X_val: List[str], y_val: List[int]):
        """Calculate weights for ensemble based on validation performance."""
        weights = {}
        
        # Traditional models
        if self.traditional_models.is_trained:
            traditional_preds = self.traditional_models.predict(X_val)
            for name, probs in traditional_preds.items():
                pred_labels = np.argmax(probs, axis=1)
                accuracy = accuracy_score(y_val, pred_labels)
                weights[f'traditional_{name}'] = accuracy
        
        # LSTM model
        if self.lstm_model.is_trained:
            lstm_probs = self.lstm_model.predict(X_val)
            lstm_pred_labels = np.argmax(lstm_probs, axis=1)
            accuracy = accuracy_score(y_val, lstm_pred_labels)
            weights['lstm'] = accuracy
        
        # Transformer model
        if self.transformer_model.is_trained:
            transformer_probs = self.transformer_model.predict(X_val)
            transformer_pred_labels = np.argmax(transformer_probs, axis=1)
            accuracy = accuracy_score(y_val, transformer_pred_labels)
            weights['transformer'] = accuracy
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}
        
        logger.info(f"Ensemble weights: {self.weights}")
    
    def predict(self, texts: List[str]) -> Dict:
        """Predict sentiment using ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        all_predictions = {}
        
        # Get predictions from all models
        if self.traditional_models.is_trained:
            traditional_preds = self.traditional_models.predict(texts)
            all_predictions.update({f'traditional_{k}': v for k, v in traditional_preds.items()})
        
        if self.lstm_model.is_trained:
            all_predictions['lstm'] = self.lstm_model.predict(texts)
        
        if self.transformer_model.is_trained:
            all_predictions['transformer'] = self.transformer_model.predict(texts)
        
        # Combine predictions using weights
        ensemble_probs = self._combine_predictions(all_predictions)
        
        # Get final predictions
        pred_labels = np.argmax(ensemble_probs, axis=1)
        pred_labels_str = self.label_encoder.inverse_transform(pred_labels)
        
        # Get confidence scores
        confidence_scores = np.max(ensemble_probs, axis=1)
        
        return {
            'predictions': pred_labels_str,
            'probabilities': ensemble_probs,
            'confidence': confidence_scores,
            'individual_predictions': all_predictions
        }
    
    def _combine_predictions(self, all_predictions: Dict) -> np.ndarray:
        """Combine predictions from all models using weights."""
        if self.config.ensemble_method == "simple":
            # Simple averaging
            combined = np.mean(list(all_predictions.values()), axis=0)
        elif self.config.ensemble_method == "weighted":
            # Weighted averaging
            combined = np.zeros_like(list(all_predictions.values())[0])
            for model_name, probs in all_predictions.items():
                weight = self.weights.get(model_name, 1.0 / len(all_predictions))
                combined += weight * probs
        else:
            # Default to simple averaging
            combined = np.mean(list(all_predictions.values()), axis=0)
        
        return combined
    
    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict:
        """Evaluate ensemble performance."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        # Get predictions
        results = self.predict(X_test)
        predictions = results['predictions']
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'average_confidence': np.mean(results['confidence'])
        }
    
    def save_model(self, filepath: str):
        """Save the trained ensemble model."""
        model_data = {
            'config': self.config,
            'label_encoder': self.label_encoder,
            'weights': self.weights,
            'is_trained': self.is_trained
        }
        
        # Save traditional models
        if self.traditional_models.is_trained:
            model_data['traditional_models'] = self.traditional_models.models
        
        # Save LSTM model
        if self.lstm_model.is_trained:
            lstm_path = filepath.replace('.pkl', '_lstm.h5')
            self.lstm_model.model.save(lstm_path)
            model_data['lstm_tokenizer'] = self.lstm_model.tokenizer
            model_data['lstm_path'] = lstm_path
        
        # Save main model data
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained ensemble model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        ensemble = cls(model_data['config'])
        ensemble.label_encoder = model_data['label_encoder']
        ensemble.weights = model_data['weights']
        ensemble.is_trained = model_data['is_trained']
        
        # Load traditional models
        if 'traditional_models' in model_data:
            ensemble.traditional_models.models = model_data['traditional_models']
            ensemble.traditional_models.is_trained = True
        
        # Load LSTM model
        if 'lstm_path' in model_data:
            ensemble.lstm_model.model = tf.keras.models.load_model(model_data['lstm_path'])
            ensemble.lstm_model.tokenizer = model_data['lstm_tokenizer']
            ensemble.lstm_model.is_trained = True
        
        # Load transformer model
        ensemble.transformer_model.load_pretrained()
        
        logger.info(f"Model loaded from {filepath}")
        return ensemble


# Example usage and testing
if __name__ == "__main__":
    # Sample data
    sample_texts = [
        "I love this product! It's amazing!",
        "This is terrible. Worst experience ever.",
        "It's okay, nothing special.",
        "Absolutely fantastic! Highly recommend!",
        "Not good at all. Very disappointed.",
        "Pretty decent, could be better.",
        "Outstanding quality and service!",
        "Meh, average at best.",
        "Horrible! Don't waste your money!",
        "Great value for money!"
    ]
    
    sample_labels = [
        "positive", "negative", "neutral", "positive", "negative",
        "neutral", "positive", "neutral", "negative", "positive"
    ]
    
    # Create and configure ensemble
    config = ModelConfig(
        max_features=5000,
        epochs=5,
        batch_size=16
    )
    
    ensemble = SentimentEnsemble(config)
    
    # Train ensemble
    print("Training ensemble...")
    training_results = ensemble.train(sample_texts, sample_labels)
    
    # Test predictions
    test_texts = [
        "This product is absolutely wonderful!",
        "I hate this so much.",
        "It's fine, I guess."
    ]
    
    print("\nMaking predictions...")
    predictions = ensemble.predict(test_texts)
    
    for i, text in enumerate(test_texts):
        pred = predictions['predictions'][i]
        conf = predictions['confidence'][i]
        print(f"Text: {text}")
        print(f"Prediction: {pred} (confidence: {conf:.3f})")
        print("-" * 50)
    
    # Save model
    ensemble.save_model("sentiment_ensemble.pkl")
    print("\nModel saved successfully!")