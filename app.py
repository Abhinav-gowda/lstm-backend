"""
Ingredient Analyzer API
Flask REST API for analyzing cosmetic product ingredients for harmfulness.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import model_loader
import config

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, origins=config.CORS_ORIGINS)


# ==================== Routes ====================

@app.route('/')
def index():
    """Root endpoint - API information"""
    return jsonify({
        'name': 'Ingredient Analyzer API',
        'version': '1.0.0',
        'description': 'API for analyzing cosmetic product ingredients for harmfulness',
        'endpoints': {
            'health': '/health',
            'model_info': '/api/model/info',
            'ingredient_lookup': '/api/ingredient/<name>',
            'predict_effect': '/api/predict/effect',
            'analyze_product': '/api/analyze/product'
        }
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader._model is not None
    })


@app.route('/api/model/info')
def model_info():
    """Get information about the loaded model"""
    try:
        info = model_loader.get_model_info()
        return jsonify({
            'success': True,
            'data': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ingredient/<name>')
def get_ingredient(name):
    """
    Look up an ingredient by name.
    
    Args:
        name: Ingredient name (URL encoded)
    
    Returns:
        JSON with ingredient information or 404 if not found
    """
    try:
        ingredient = model_loader.get_ingredient_info(name)
        
        if ingredient:
            return jsonify({
                'success': True,
                'data': ingredient
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Ingredient "{name}" not found in database'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict/effect', methods=['POST'])
def predict_effect():
    """
    Predict harm score for an ingredient based on its effect description.
    
    Request body:
        {
            "effect": "string describing the ingredient's effect on human body"
        }
    
    Returns:
        JSON with predicted harm score
    """
    try:
        data = request.get_json()
        
        if not data or 'effect' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "effect" field in request body'
            }), 400
        
        effect_text = data['effect']
        
        # Predict harm score
        harm_score = model_loader.predict_harm_score(effect_text)
        
        return jsonify({
            'success': True,
            'data': {
                'effect': effect_text,
                'predicted_harm_score': round(harm_score, 2)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze/product', methods=['POST'])
def analyze_product():
    """
    Analyze a complete product with multiple ingredients.
    
    Request body:
        {
            "ingredients": [
                {"name": "Water", "effect": "..."},
                {"name": "Glycerin", "effect": "..."},
                ...
            ]
        }
    
    Returns:
        JSON with analysis results for each ingredient and overall product risk
    """
    try:
        data = request.get_json()
        
        if not data or 'ingredients' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "ingredients" field in request body'
            }), 400
        
        ingredients = data['ingredients']
        
        if not ingredients:
            return jsonify({
                'success': False,
                'error': 'Ingredients list cannot be empty'
            }), 400
        
        # Analyze each ingredient
        analyzed_ingredients = []
        harm_scores = []
        
        for ingredient in ingredients:
            name = ingredient.get('name', '')
            effect = ingredient.get('effect', '')
            
            # Try to get from database first
            db_ingredient = model_loader.get_ingredient_info(name)
            
            if db_ingredient:
                harm_score = db_ingredient['harm_score']
                source = 'database'
            elif effect:
                # Predict using model
                harm_score = model_loader.predict_harm_score(effect)
                source = 'model_prediction'
            else:
                # Skip if no effect provided and not in database
                harm_score = None
                source = 'not_found'
            
            if harm_score is not None:
                harm_scores.append(harm_score)
            
            analyzed_ingredients.append({
                'name': name,
                'effect': effect,
                'harm_score': round(harm_score, 2) if harm_score else None,
                'source': source
            })
        
        # Calculate product scores
        if harm_scores:
            scores = model_loader.calculate_product_scores(harm_scores)
            risk_classification = model_loader.classify_product_risk(scores['final_score'])
        else:
            scores = {
                'average_score': 0,
                'maximum_score': 0,
                'final_score': 0
            }
            risk_classification = 'Unknown'
        
        return jsonify({
            'success': True,
            'data': {
                'ingredients': analyzed_ingredients,
                'product_scores': {
                    'average_score': round(scores['average_score'], 2),
                    'maximum_score': round(scores['maximum_score'], 2),
                    'final_score': round(scores['final_score'], 2)
                },
                'risk_classification': risk_classification
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ==================== Main ====================

if __name__ == '__main__':
    # Load model and data on startup
    print("Loading model and data...")
    try:
        model_loader.load_model_and_tokenizer()
        model_loader.load_ingredients_data()
        print("All resources loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load all resources: {e}")
        print("Some endpoints may not work until model files are available.")
    
    # Run the app
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG_MODE
    )
