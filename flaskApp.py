from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved Random Forest model
with open('random_forest_model.pkl', 'rb') as f:
    trained_trees = pickle.load(f)

# Define the prediction function
def predict(trees, input_data):
    # Helper function to predict using the random forest
    def bagging_predict(trees, row):
        predictions = [predict_tree(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)
    
    # Helper function to predict with a single tree
    def predict_tree(node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict_tree(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict_tree(node['right'], row)
            else:
                return node['right']
    
    return bagging_predict(trees, input_data)

@app.route('/')
def home():
    return "Random Forest Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Parse input JSON
        input_json = request.json
        
        # Extract features in the correct order
        feature_order = ['Rainfall', 'Avg Temp', 'Relative Humidity', 'Sand %', 
                         'PH Level', 'Phosohorus', 'Potassium', 'Clay %']
        input_data = [input_json[feature] for feature in feature_order]
        
        # Ensure input is a NumPy array
        input_data = np.array(input_data).reshape(1, -1)
        
        # Make a prediction
        prediction = predict(trained_trees, input_data[0])
        
        # Return the result
        return jsonify({'prediction': int(prediction)})
    except KeyError as e:
        return jsonify({'error': f"Missing feature: {str(e)}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
