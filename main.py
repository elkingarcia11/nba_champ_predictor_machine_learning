import csv
from datetime import datetime
import os
import random

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import tensorflow as tf


def import_data(excluded_year):
    """
    Import either sports data for multiple years, excluding a specified year.

    Parameters:
    - excluded_year (int): The year to exclude from the data.

    Returns:
    - X_year (list): List of advanced data arrays for each year (excluding `excluded_year`).
    - y (list): List of target values corresponding to the teams and years.
    """

    # Training set: Every year
    X_year = []

    # Target values
    y = []

    # End Year
    end_year = datetime.now().year

    csv_path = ""

    # Start year for data
    start_year = 2000

    for year in range(start_year, end_year):
        if year != excluded_year:
            csv_path = f"data/{str(year)}.csv"

            if os.path.exists(csv_path):  # Check if the file exists
                with open(csv_path, 'r') as file:
                    csv_reader = csv.reader(file)
                    next(csv_reader)  # Skip header row
                    for row in csv_reader:
                        # Accessing the last element directly
                        y.append(row[-1])
                        X_year.append(row[3:-1])
    return X_year, y


def import_prediction(excluded_year):
    """
    Import data for prediction from a specified year.

    Parameters:
    - excluded_year (int): The year for which data is to be imported for prediction.

    Returns:
    - float_X (numpy.ndarray): Data array for prediction, converted to float.
    """

    X = []
    csv_path = f"data/{str(excluded_year)}.csv"

    if os.path.exists(csv_path):
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                X.append(row[3:-1])

    if not X:
        return None
    Xn = np.array(X)
    float_X = Xn.astype(float)
    return float_X


def train_model(excluded_year):
    """
    Train a neural network model using the imported data.

    Parameters:
    - excluded_year (int): The year to exclude from the training data.

    Returns:
    - Trained Keras Sequential model.
    """

    X, y = import_data(excluded_year)
    # Convert data and target values to NumPy arrays and ensure they are in float format
    Xn = np.array(X)
    Yn = np.array(y)
    float_X = Xn.astype(float)
    float_Y = Yn.astype(float)

    # Normalize the input data using TensorFlow's Normalization layer
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(float_X)  # learns mean, variance

    # Generate a random 64-bit integer
    random_int64 = random.getrandbits(64)

    # Set a random seed for reproducibility in TensorFlow
    tf.random.set_seed(random_int64)

    # Define a Sequential model with two Dense layers
    model = Sequential(
        [
            Dense(16, activation='relu'),
            # Ideal for classification problems 0/1
            Dense(1, activation='sigmoid'),
        ]
    )

    # Compile the model with a binary cross-entropy loss function and Adam optimizer
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        # Learning rate controls the size of steps taken during optimization process
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )

    # Train the model using the training data and target values

    model.fit(
        float_X, float_Y,
        # epochs = how many times training set is passed through NN during training
        epochs=300
    )

    return model


def get_user_input():
    """
    Prompt the user for input regarding the prediction year and whether to use stats.

    Returns:
    - prediction_year (int): The year for which data is to be imported for prediction.
    """

    while True:
        try:
            prediction_year = int(input("Enter the year for prediction: "))
            break
        except ValueError:
            print("Please enter a valid year.")

    return prediction_year


def get_predictions(prediction_year):
    """
    Return the predictions based on user input.

    Parameters:
    - prediction_year (int): The year for which data is to be imported for prediction.

    Returns:
    - predictions (numpy.ndarray): An array of expected target values for every team in the excluded year.
    """

    model = train_model(prediction_year)

    x = import_prediction(prediction_year)

    if x is not None:
        predictions = model.predict(x)

        # Display the predictions
        # print("predictions = \n", predictions)
        return predictions
    else:
        print("No data available for prediction year.")


def print_best_team(prediction_year, predictions):
    """
    Prints the team with the maximum predicted value for a given prediction year.

    Parameters:
        prediction_year (int): The year for which predictions are made.
        predictions (list): A list of predicted values for each team.

    Returns:
        None
    """

    # Specify the file path
    csv_file_path = f"data/{str(prediction_year)}.csv"

    # Initialize an empty list to store the data
    data = []

    # Open the CSV file
    with open(csv_file_path, newline='') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Append each row to the data list
            data.append(row)

    # Find the index of the maximum value
    max_index = np.argmax(predictions)
    # Extract the team name
    team_name = data[max_index+1][2]

    print("Team with the maximum value:", team_name)


# Get user input
prediction_year = get_user_input()
predictions = get_predictions(prediction_year)
print_best_team(prediction_year, predictions)
