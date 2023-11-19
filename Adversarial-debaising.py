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
#
#
#
# import tensorflow as tf
#
# # Define the model
# def make_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(10, activation='softmax')
#     ])
#     return model
#
# # Define the secondary model
# def make_debiasing_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#     return model
#
# # Define the loss function
# def loss_fn(y_true, y_pred):
#     return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)
#
# # Define the loss function for the secondary model
# def debiasing_loss_fn(y_true, y_pred):
#     return tf.keras.losses.binary_crossentropy(y_true, y_pred)
#
# # Define the optimizer
# optimizer = tf.keras.optimizers.Adam(0.001)
#
# # Define the optimizer for the secondary model
# debiasing_optimizer = tf.keras.optimizers.Adam(0.001)
#
# # Define the metrics
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
#
# # Define the metrics for the secondary model
# debiasing_train_loss = tf.keras.metrics.Mean(name='debiasing_train_loss')
# debiasing_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='debiasing_train_accuracy')
#
# # Define the training step
# @tf.function
# def train_step(inputs, labels, debiasing_labels):
#     with tf.GradientTape() as tape, tf.GradientTape() as debiasing_tape:
#         predictions = model(inputs)
#         debiasing_predictions = debiasing_model(inputs)
#
#         loss = loss_fn(labels, predictions)
#         debiasing_loss = debiasing_loss_fn(debiasing_labels, debiasing_predictions)
#
#     gradients = tape.gradient(loss, model.trainable_variables)
#     debiasing_gradients = debiasing_tape.gradient(debiasing_loss, debiasing_model.trainable_variables)
#
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     debiasing_optimizer.apply_gradients(zip(debiasing_gradients, debiasing_model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#     debiasing_train_loss(debiasing_loss)
#     debiasing_train_accuracy(debiasing_labels, debiasing_predictions)
#
# # Define the training loop
# EPOCHS = 5
#
# for epoch in range(EPOCHS):
#     for images, labels in train_ds:
#         debiasing_labels = tf.random.uniform(shape=(labels.shape[0], 1), minval=0, maxval=2, dtype=tf.int32)
#         train_step(images, labels, debiasing_labels)
#
#     template = 'Epoch {}, Loss: {}, Accuracy: {}, De-Biasing Loss: {}, De-Biasing Accuracy: {}'
#     print(template.format(epoch+1,
#                           train_loss.result(),
#                           train_accuracy.result()*100,
#                           debiasing_train_loss.result(),
#                           debiasing_train_accuracy.result()*100))
#
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     debiasing_train_loss.reset_states()
#     debiasing_train_accuracy.reset_states()
#
# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(test_ds)
# print('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_accuracy*100))

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
print(compas.cov())
print(compas.corr())