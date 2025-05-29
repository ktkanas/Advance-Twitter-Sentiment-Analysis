"""
Advanced Text Preprocessing Pipeline for Sentiment Analysis
Author: Your Name
Description: Professional-grade text preprocessing with comprehensive cleaning,
            normalization, and feature extraction capabilities.
"""

import re
import string
import unicodedata
from typing import List, Dict, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import emoji
from textblob import TextBlob
from spellchecker import SpellChecker
import contractions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration class for text preprocessing parameters."""
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = False
    handle_emojis: str = 'convert'  # 'remove', 'convert', 'keep'
    expand_contractions: bool = True
    correct_spelling: bool = True
    remove_punctuation: bool = True
    convert_to_lowercase: bool = True
    remove_stopwords: bool = True
    stemming: bool = False
    lemmatization: bool = True
    min_word_length: int = 2
    max_word_length: int = 15
    remove_numbers: bool = False
    preserve_case_for_sentiment: bool = False


class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing pipeline for social media text analysis.
    
    Features:
    - Intelligent emoji handling with sentiment preservation
    - Context-aware spell checking
    - Advanced tokenization for social media text
    - Configurable preprocessing steps
    - Batch processing capabilities
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the preprocessor with configuration."""
        self.config = config or PreprocessingConfig()
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize NLTK components and other resources."""
        try:
            # Download required NLTK data
            nltk_downloads = [
                'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'vader_lexicon', 'omw-1.4'
            ]
            for item in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{item}')
                except LookupError:
                    nltk.download(item, quiet=True)
            
            # Initialize components
            self.tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True)
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            self.spell_checker = SpellChecker()
            
            # Load stopwords
            self.stop_words = set(stopwords.words('english'))
            
            # Custom stopwords for social media
            social_media_stopwords = {
                'rt', 'via', 'amp', 'get', 'got', 'go', 'going', 'gone',
                'today', 'tomorrow', 'yesterday', 'day', 'time', 'year'
            }
            self.stop_words.update(social_media_stopwords)
            
            # Emoji patterns
            self.emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map
                "\U0001F1E0-\U0001F1FF"  # flags
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+", 
                flags=re.UNICODE
            )
            
            logger.info("Text preprocessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing preprocessor: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Main preprocessing function that applies all cleaning steps.
        
        Args:
            text (str): Raw text to be cleaned
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Store original for reference
            original_text = text
            
            # Step 1: Basic cleaning
            text = self._basic_cleaning(text)
            
            # Step 2: Handle URLs
            if self.config.remove_urls:
                text = self._remove_urls(text)
            
            # Step 3: Handle mentions and hashtags
            if self.config.remove_mentions:
                text = self._remove_mentions(text)
            if self.config.remove_hashtags:
                text = self._remove_hashtags(text)
            
            # Step 4: Handle emojis
            text = self._handle_emojis(text)
            
            # Step 5: Expand contractions
            if self.config.expand_contractions:
                text = self._expand_contractions(text)
            
            # Step 6: Handle case
            if self.config.convert_to_lowercase:
                text = text.lower()
            
            # Step 7: Tokenization
            tokens = self._tokenize(text)
            
            # Step 8: Spell checking
            if self.config.correct_spelling:
                tokens = self._correct_spelling(tokens)
            
            # Step 9: Remove punctuation
            if self.config.remove_punctuation:
                tokens = self._remove_punctuation(tokens)
            
            # Step 10: Filter by length
            tokens = self._filter_by_length(tokens)
            
            # Step 11: Remove stopwords
            if self.config.remove_stopwords:
                tokens = self._remove_stopwords(tokens)
            
            # Step 12: Stemming or Lemmatization
            if self.config.lemmatization:
                tokens = self._lemmatize_tokens(tokens)
            elif self.config.stemming:
                tokens = self._stem_tokens(tokens)
            
            # Step 13: Remove numbers if specified
            if self.config.remove_numbers:
                tokens = [token for token in tokens if not token.isdigit()]
            
            # Join tokens back to text
            cleaned_text = " ".join(tokens)
            
            # Final validation
            if not cleaned_text.strip():
                logger.warning(f"Text became empty after preprocessing: {original_text[:50]}...")
                return original_text  # Return original if cleaning results in empty string
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return text  # Return original text if error occurs
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning operations."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        return text.strip()
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        text = url_pattern.sub('', text)
        
        # Remove www links
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text
    
    def _remove_mentions(self, text: str) -> str:
        """Remove @mentions from text."""
        return re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    def _remove_hashtags(self, text: str) -> str:
        """Remove hashtags from text."""
        return re.sub(r'#[A-Za-z0-9_]+', '', text)
    
    def _handle_emojis(self, text: str) -> str:
        """Handle emojis based on configuration."""
        if self.config.handle_emojis == 'remove':
            return self.emoji_pattern.sub('', text)
        elif self.config.handle_emojis == 'convert':
            return emoji.demojize(text, delimiters=(" ", " "))
        else:  # keep
            return text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        try:
            return contractions.fix(text)
        except Exception:
            # Fallback to basic contractions
            contractions_dict = {
                "won't": "will not", "can't": "cannot", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will",
                "'d": " would", "'m": " am", "it's": "it is",
                "that's": "that is", "what's": "what is"
            }
            for contraction, expansion in contractions_dict.items():
                text = text.replace(contraction, expansion)
            return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using appropriate tokenizer."""
        tokens = self.tweet_tokenizer.tokenize(text)
        return [token for token in tokens if token and not token.isspace()]
    
    def _correct_spelling(self, tokens: List[str]) -> List[str]:
        """Correct spelling of tokens."""
        corrected_tokens = []
        for token in tokens:
            if token.isalpha() and len(token) > 2:
                # Only correct if the word is likely misspelled
                if token.lower() not in self.spell_checker:
                    # Get the most likely correction
                    correction = self.spell_checker.correction(token.lower())
                    if correction and correction != token.lower():
                        corrected_tokens.append(correction)
                    else:
                        corrected_tokens.append(token)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)
        return corrected_tokens
    
    def _remove_punctuation(self, tokens: List[str]) -> List[str]:
        """Remove punctuation from tokens."""
        return [token for token in tokens if not all(c in string.punctuation for c in token)]
    
    def _filter_by_length(self, tokens: List[str]) -> List[str]:
        """Filter tokens by length."""
        return [
            token for token in tokens 
            if self.config.min_word_length <= len(token) <= self.config.max_word_length
        ]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def _stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens with POS tagging."""
        # Get POS tags
        pos_tags = pos_tag(tokens)
        
        lemmatized = []
        for token, pos in pos_tags:
            # Convert POS tag to WordNet format
            wordnet_pos = self._get_wordnet_pos(pos)
            if wordnet_pos:
                lemmatized.append(self.lemmatizer.lemmatize(token, wordnet_pos))
            else:
                lemmatized.append(self.lemmatizer.lemmatize(token))
        
        return lemmatized
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert TreeBank POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return None
    
    def batch_process(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """
        Process multiple texts in batch.
        
        Args:
            texts (List[str]): List of texts to process
            show_progress (bool): Whether to show progress bar
            
        Returns:
            List[str]: List of processed texts
        """
        processed_texts = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Processing texts")
            except ImportError:
                iterator = texts
                logger.info(f"Processing {len(texts)} texts...")
        else:
            iterator = texts
        
        for text in iterator:
            processed_texts.append(self.clean_text(text))
        
        return processed_texts
    
    def get_preprocessing_stats(self, original_texts: List[str], processed_texts: List[str]) -> Dict:
        """
        Get statistics about the preprocessing operation.
        
        Args:
            original_texts (List[str]): Original texts
            processed_texts (List[str]): Processed texts
            
        Returns:
            Dict: Preprocessing statistics
        """
        stats = {
            'total_texts': len(original_texts),
            'avg_original_length': np.mean([len(text.split()) for text in original_texts]),
            'avg_processed_length': np.mean([len(text.split()) for text in processed_texts]),
            'reduction_ratio': 1 - (
                np.mean([len(text.split()) for text in processed_texts]) / 
                np.mean([len(text.split()) for text in original_texts])
            ),
            'empty_results': sum(1 for text in processed_texts if not text.strip()),
            'config': self.config.__dict__
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Sample tweets for testing
    sample_tweets = [
        "I absolutely LOVE this new product! 😍 It's amazing!!! #ProductLove @company",
        "This is the worst service I've ever experienced... won't recommend 😡",
        "Can't believe how gr8 this is! U should definitely try it out! https://example.com",
        "Meh... it's okay I guess. Nothing special tbh 🤷‍♀️",
        "AMAZING!!!! Best purchase ever!!! ❤️❤️❤️ #Happy #Satisfied"
    ]
    
    # Initialize preprocessor with custom config
    config = PreprocessingConfig(
        handle_emojis='convert',
        correct_spelling=True,
        lemmatization=True
    )
    
    preprocessor = AdvancedTextPreprocessor(config)
    
    # Process samples
    print("Original vs Processed Tweets:")
    print("=" * 80)
    
    processed_tweets = []
    for i, tweet in enumerate(sample_tweets, 1):
        processed = preprocessor.clean_text(tweet)
        processed_tweets.append(processed)
        print(f"Tweet {i}:")
        print(f"Original:  {tweet}")
        print(f"Processed: {processed}")
        print("-" * 40)
    
    # Get stats
    stats = preprocessor.get_preprocessing_stats(sample_tweets, processed_tweets)
    print("\nPreprocessing Statistics:")
    print("=" * 30)
    for key, value in stats.items():
        if key != 'config':
            print(f"{key}: {value}")
