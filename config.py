import os

# Configuration settings for the Ingredient Analyzer API

# Model paths
MODEL_PATH = os.environ.get('MODEL_PATH', 'lstm_model.keras')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', 'tokenizer.pkl')
INGREDIENTS_DATA_PATH = os.environ.get('INGREDIENTS_DATA_PATH', 'ingredient_effects.csv')

# Model configuration
MAX_SEQUENCE_LENGTH = 12  # From notebook: padded to 12
EMBEDDING_DIM = 100
VOCAB_SIZE = None  # Will be set when tokenizer is loaded

# API Configuration
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', 5000))
DEBUG_MODE = os.environ.get('DEBUG', 'True').lower() == 'true'

# Risk classification thresholds
RISK_THRESHOLDS = {
    'safe': 5,
    'moderate_risk': 7,
    'harmful': float('inf')
}

# CORS settings
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
