from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load your trained model
model = joblib.load('spam_model.joblib')
print("Model loaded:", model)  # Add this to check loading

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data['message']
        
        # Get prediction (1 for ham, 0 for spam)
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        confidence = probabilities.max()
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    
    