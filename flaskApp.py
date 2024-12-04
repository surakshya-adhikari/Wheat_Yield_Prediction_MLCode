from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved Random Forest model
with open('random_forest_regression_model3.pkl', 'rb') as f:
    trained_trees = pickle.load(f)

# Helper function to predict with a single tree
def predict_tree(node, row):
    if isinstance(node, (int, float)):  # Leaf node
        return node
    if row[node['feature_idx']] <= node['threshold']:  # Go left
        return predict_tree(node['left'], row)
    else:  # Go right
        return predict_tree(node['right'], row)

# Helper function to predict using the random forest (bagging)
def predict(trees, input_data):
    predictions = [predict_tree(tree, input_data) for tree in trees]
    return np.mean(predictions)  # For regression, return the average prediction

@app.route('/')
def home():
    return "Random Forest Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Parse input JSON
        input_json = request.json
        
        # Extract features in the correct order
        feature_order = ['Rainfall', 'AvgTemp', 'RelativeHumidity','SoilTemp', 'Sand', 
                         'PHLevel', 'Phosohorus', 'Potassium', 'Clay','ProductionArea']
        input_data = [input_json[feature] for feature in feature_order]
        
        # Ensure input is a NumPy array
        input_data = np.array(input_data).reshape(1, -1)
        
        # Make a prediction
        prediction = predict(trained_trees, input_data[0])
        
        data = request.get_json()
        print("Received data: ", data)  # Debugging line to check the input

    

        # Return the result
        print("prediction",prediction)
        return jsonify({'prediction': prediction})
    except KeyError as e:
        return jsonify({'error': f"Missing feature: {str(e)}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
