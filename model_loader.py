"""
Model Loader Utility
Loads the trained LSTM model, tokenizer, and ingredient data for predictions.
"""

import pickle
import re
import string
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import config

# Global variables to store loaded resources
_model = None
_tokenizer = None
_ingredients_df = None
_max_sequence_length = None


def clean_text(text):
    """Converts text to lowercase, removes punctuation, and strips whitespace."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = text.strip()
    return text


def load_model_and_tokenizer():
    """
    Load the trained LSTM model and tokenizer.
    Returns: (model, tokenizer, vocab_size)
    """
    global _model, _tokenizer, _max_sequence_length
    
    if _model is None:
        print(f"Loading model from {config.MODEL_PATH}...")
        _model = load_model(config.MODEL_PATH)
        print("Model loaded successfully!")
    
    if _tokenizer is None:
        print(f"Loading tokenizer from {config.TOKENIZER_PATH}...")
        with open(config.TOKENIZER_PATH, 'rb') as f:
            _tokenizer = pickle.load(f)
        print("Tokenizer loaded successfully!")
    
    vocab_size = len(_tokenizer.word_index) + 1
    _max_sequence_length = config.MAX_SEQUENCE_LENGTH
    
    return _model, _tokenizer, vocab_size


def load_ingredients_data():
    """
    Load the ingredient effects CSV data.
    Returns: DataFrame with ingredient data
    """
    global _ingredients_df
    
    if _ingredients_df is None:
        print(f"Loading ingredients data from {config.INGREDIENTS_DATA_PATH}...")
        _ingredients_df = pd.read_csv(config.INGREDIENTS_DATA_PATH)
        print(f"Loaded {_len(_ingredients_df)} ingredients!")
    
    return _ingredients_df


def get_ingredient_info(ingredient_name):
    """
    Get ingredient information from the database.
    Returns: dict with ingredient details or None if not found
    """
    if _ingredients_df is None:
        load_ingredients_data()
    
    # Search for ingredient (case-insensitive)
    match = _ingredients_df[_ingredients_df['Ingredient_Name'].str.lower() == ingredient_name.lower()]
    
    if not match.empty:
        row = match.iloc[0]
        return {
            'name': row['Ingredient_Name'],
            'harm_score': float(row['Harmfulness_Score']),
            'effect': row['Effect_On_Human_Body']
        }
    
    return None


def predict_harm_score(raw_effect_text, model=None, tokenizer=None):
    """
    Predict the harm score for a given raw effect text.
    
    Args:
        raw_effect_text: Raw text describing the ingredient's effect on human body
        model: Optional model instance (will load if not provided)
        tokenizer: Optional tokenizer instance (will load if not provided)
    
    Returns:
        float: Predicted harm score (clamped between 1 and 10)
    """
    if model is None or tokenizer is None:
        model, tokenizer, _ = load_model_and_tokenizer()
    
    # Clean the raw text
    cleaned_text = clean_text(raw_effect_text)
    
    # Tokenize the cleaned text
    encoded_text = tokenizer.texts_to_sequences([cleaned_text])
    
    # Pad the sequence
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded_text = pad_sequences(encoded_text, maxlen=_max_sequence_length, padding='post')
    
    # Predict the harm score
    predicted_score = model.predict(padded_text, verbose=0)[0][0]
    
    # Clamp the score between 1 and 10
    clamped_score = np.clip(predicted_score, 1, 10)
    
    return float(clamped_score)


def calculate_product_scores(ingredient_harm_scores):
    """
    Calculate average, maximum, and final scores for a product.
    
    Args:
        ingredient_harm_scores: List of individual ingredient harm scores
    
    Returns:
        dict with average_score, maximum_score, final_score
    """
    if not ingredient_harm_scores:
        return {
            'average_score': 0,
            'maximum_score': 0,
            'final_score': 0
        }
    
    average_score = np.mean(ingredient_harm_scores)
    maximum_score = np.max(ingredient_harm_scores)
    final_score = (average_score + maximum_score) / 2
    
    return {
        'average_score': float(average_score),
        'maximum_score': float(maximum_score),
        'final_score': float(final_score)
    }


def classify_product_risk(final_score):
    """
    Classify product risk based on the final harm score.
    
    Args:
        final_score: The calculated final score
    
    Returns:
        str: Risk classification ('Safe', 'Moderate Risk', or 'Harmful')
    """
    if final_score <= 5:
        return 'Safe'
    elif final_score <= 7:
        return 'Moderate Risk'
    else:
        return 'Harmful'


def get_model_info():
    """Get information about the loaded model."""
    if _model is None:
        load_model_and_tokenizer()
    
    return {
        'model_path': config.MODEL_PATH,
        'tokenizer_path': config.TOKENIZER_PATH,
        'max_sequence_length': _max_sequence_length,
        'embedding_dim': config.EMBEDDING_DIM
    }
