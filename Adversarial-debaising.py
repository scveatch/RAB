# # Adversarial De-Biasing
# #
# # In this approach, there are two (or more) models, one which is responsible for encoding information,
# # and a second which is responsible for removing latent representations from the model, i.e., racial information.
# # More generally, if there is some sensitive element "Z" in your data, we would train our secondary variable to
# # predict against a second, noncorrelated variable "Y" (which I may think about the applications of introducing
# # to the dataset as a random element) and NOT predict on Z. In this, effectively, the secondary model would
# # prevent the main model from learning from any parameters in Z in a significant way, while not removing data
# # that may otherwise be highly correlated.
#
# Adversarial De-Biasing
# This approach requires a fundamental change to the structure of the model, and therefore is not necessarily
# the best approach. However, it is a valid one that I believe will likely have some positive impact. Think
# of this model as a neural network with one head towards predicting y, some significant aspect of the data,
# and another towards predicting z, some biased variable (race or age, for example). Consider G(x) to be the
# shared embedding for the layer. Let f (x) be some prediction function executed by the model, such that
# y = f (g(x)). Additionally, set a to be an adversarial layer, such that z = a(g(x)). The unification of these
# functions is done through some negative gradient defined Nλ, where λ is a tunable hyperparameter.
#
# Ly and Lz are the respective loss functions for y and z. Additionally, the secondary approach could be
# considering the adversarial model to be wholly separate, which may reduce complexities in gradient updates.
# The adversarial model is solely focused on minimizing its own loss, ergo, its gradient weights U are updated
# by ∆uLa. The predictor’s weights, however, are calculated by
# ∆wLp − proj∆w La ∆wLp − α∆wLa
# where α is a hyperparameter controlling accuracy/debiasing tradeoff. We could also seek to maximize the
# entropy of the predictive model, effectively playing a fully zero-sum game of predictive accuracy vs. bias
# removal.

import os
import pandas as pd
import tensorflow as tf

root = os.path.dirname(os.path.realpath('Adversarial-debaising.py'))

# read in compas-scores-raw.csv from /Datasets/
compas = pd.read_csv(root + '/Datasets/compas-scores-raw.csv') # recidivism data with bias in gender and race in reation to recidivism scores

# get col headers
# print(compas.columns) # ['Person_ID', 'AssessmentID', 'Case_ID', 'Agency_Text', 'LastName', 'FirstName', 'MiddleName', 'Sex_Code_Text', 'Ethnic_Code_Text', 'DateOfBirth', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason', 'Language', 'LegalStatus', 'CustodyStatus', 'MaritalStatus', 'Screening_Date', 'RecSupervisionLevel', 'RecSupervisionLevelText', 'Scale_ID', 'DisplayText', 'RawScore', 'DecileScore', 'ScoreText', 'AssessmentType', 'IsCompleted', 'IsDeleted'

# filter for DisplayText == 'Risk of Recidivism'
# compas = compas[compas['DisplayText'] == 'Risk of Recidivism']

# drop the columns that are not needed
compas = compas.drop(['Person_ID', 'AssessmentID', 'Case_ID', 'Agency_Text', 'LastName', 'FirstName', 'MiddleName', 'DateOfBirth', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason', 'Language', 'LegalStatus', 'CustodyStatus', 'MaritalStatus', 'Screening_Date', 'RecSupervisionLevel', 'RecSupervisionLevelText', 'Scale_ID', 'DisplayText', 'RawScore', 'ScoreText', 'AssessmentType', 'IsCompleted', 'IsDeleted'], axis=1)

# drop the rows that are not needed
compas = compas.dropna()
# print(compas.columns)
# convert categorical variables to numerical, for the columns Sex_Code_Text and Ethnic_Code_Text
from sklearn.preprocessing import LabelEncoder
le1, le2 = LabelEncoder(), LabelEncoder()  # instantiate the encoder
le1.fit(pd.unique(compas['Sex_Code_Text']))  # fit the encoder to the unique values in the column
# convert the column to numerical
compas['Sex_Code_Text'] = le1.transform(compas['Sex_Code_Text'])
le2.fit(pd.unique(compas['Ethnic_Code_Text']))  # fit the encoder to the unique values in the column
# convert the column to numerical
compas['Ethnic_Code_Text'] = le2.transform(compas['Ethnic_Code_Text'])



# print the first 5 rows
# print(compas.head())




# find the covariance and collinearity between the variables Ethic_Code_Text, Sex_Code_Text, and DecileScore for DisplayText == 'Risk of Recidivism'
# covariance measures the relationship between two variables, collinearity measures the relationship between multiple variables
print(compas.cov()) # a high score indicates a high covariance between the variables, vice versa for a low score
print(compas.corr()) # a high score indicates a high collinearity between the variables, vice versa for a low score

# sample the data to get a balanced dataset for training
df_train = compas.sample(frac=0.8, random_state=0)
df_test = compas.drop(df_train.index)

# split the data into features and labels
train_features = df_train.copy()
test_features = df_test.copy()
train_labels = train_features.pop('DecileScore')
test_labels = test_features.pop('DecileScore')

print(train_features.head())
print(train_labels.head())

# normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# convert the data to tensors
train_features = tf.convert_to_tensor(train_features, dtype=tf.float32)
test_features = tf.convert_to_tensor(test_features, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)

# define the model
def make_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model
model = make_model()

# define the loss function
def loss_fn(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)

# define the optimizer
optimizer = tf.keras.optimizers.Adam(0.001)

# define the metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

# define the training step
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# define the training loop
EPOCHS = 5

for epoch in range(EPOCHS):
    for x, y in zip(train_features, train_labels):
        train_step(x, y)

    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100))

# define the test metrics
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

# define the test step
@tf.function
def test_step(inputs, labels):
    predictions = model(inputs)
    t_loss = loss_fn(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

# define the test loop
for x, y in zip(test_features, test_labels):
    test_step(x, y)
