import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os

# Load COMPAS dataset
# Make sure to preprocess the data accordingly (handle missing values, encode categorical variables, etc.)
# For simplicity, assume `X` contains features and `y` contains the target variable (recidivism)
root = os.path.dirname(os.path.realpath('Adversarial-debaising.py'))

# read in compas-scores-raw.csv from /Datasets/
compas = pd.read_csv(root + '/Datasets/compas-scores-raw.csv') # recidivism data with bias in gender and race in reation to recidivism scores



# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(compas, compas['DecileScore'], test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model architecture
def create_model(input_dim, output_dim):
    model_input = tf.keras.Input(shape=(input_dim,))

    # Main task model
    main_task = layers.Dense(64, activation='relu')(model_input)
    main_task = layers.Dense(output_dim, activation='sigmoid', name='main_task')(main_task)

    # Adversary model
    adversary = layers.Dense(32, activation='relu')(model_input)
    adversary = layers.Dense(2, activation='softmax', name='adversary')(adversary)

    model = tf.keras.Model(inputs=model_input, outputs=[main_task, adversary])
    return model

# Instantiate the model
input_dim = X_train.shape[1]
output_dim = 1  # Assuming binary classification for recidivism
model = create_model(input_dim, output_dim)

# Define custom loss functions
def main_task_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def adversary_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

# Compile the model
model.compile(optimizer='adam',
              loss={'main_task': main_task_loss, 'adversary': adversary_loss},
              metrics={'main_task': 'accuracy', 'adversary': 'accuracy'})

# Train the model
model.fit(X_train, {'main_task': y_train, 'adversary': y_train}, epochs=10, batch_size=32)

# Evaluate the model
y_pred, _ = model.predict(X_test)
y_pred_main_task = (y_pred['main_task'] > 0.5).astype(int)

# Evaluate the accuracy and fairness metrics
accuracy = accuracy_score(y_test, y_pred_main_task)
print(f"Accuracy: {accuracy}")

# Report classification metrics
print("Classification Report:\n", classification_report(y_test, y_pred_main_task))