# NBA Champion Predictor - Machine Learning Model

## Overview

A Python script designed to forecast NBA champion by leveraging historical data. Powered by TensorFlow and Keras, the script employs a neural network. The dataset encompasses various team attributes, including performance metrics like field goals attempted and percentages, three-pointers made and attempted, among others. The target variable indicates team success within a specific year, denoting 0 for non-championship years and 1 for championship years.

## Files

- `binary_champ.py`: The main Python script that imports data, trains the model, and makes predictions.
- `data/`: Directory containing CSV files with historical team data.
- `data/simple/`: Directory containing CSV files with simplified team data.
- `data/advanced/`: Directory containing CSV files with advanced team data.

## Dependencies

- csv
- random
- numpy
- tensorflow

### Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the required dependencies installed.
2. Adjust the parameters in the script as needed (e.g., `latest_year`, `excluded_year`, etc.).
3. Run the script using:

   ```bash
   python binary_champ.py
   ```

4. The script will import training data, train the neural network, and make predictions for a specified year.

## Functions

### `import_data(latest_year, excluded_year)`

Import training data and target values from multiple years, excluding a specified year.

### `import_random_data(latest_year, excluded_year)`

Import training data with random samples and target values, excluding a specified year.

### `import_prediction(excluded_year)`

Import data for prediction from a specified year.

### `import_advanced_prediction(excluded_year)`

Import advanced data for prediction from a specified year.

### `import_advanced_data(latest_year, excluded_year)`

Import advanced training data and target values from multiple years, excluding a specified year.

## Data Format

- `X_year`: List of data arrays for each year (excluding the excluded_year).
- `y`: List of target values corresponding to the teams and years.

## Model Training

1. The script imports data using the provided functions.
2. The data is converted to NumPy arrays and normalized using TensorFlow's Normalization layer.
3. A Sequential model is defined with two Dense layers.
4. The model is compiled with a binary cross-entropy loss function and Adam optimizer.
5. The model is trained using the training data and target values.

## Example Usage

```python
# Example Usage:
X, y = import_advanced_data(2024, 2009)

# ... (model training)

# Import advanced data for prediction from a specified year
x_test = import_advanced_prediction("2009")

# Make predictions using the trained model
predictions = model.predict(x_test)

# Display the predictions
print("predictions = \n", predictions)
```

Feel free to modify the script and parameters based on your specific use case and dataset.
# nba_champ_predictor_machine_learning
