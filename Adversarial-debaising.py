# Adversarial De-Biasing
#
# In this approach, there are two (or more) models, one which is responsible for encoding information,
# and a second which is responsible for removing latent representations from the model, i.e., racial information.
# More generally, if there is some sensitive element "Z" in your data, we would train our secondary variable to
# predict against a second, noncorrelated variable "Y" (which I may think about the applications of introducing
# to the dataset as a random element) and NOT predict on Z. In this, effectively, the secondary model would
# prevent the main model from learning from any parameters in Z in a significant way, while not removing data
# that may otherwise be highly correlated.




import tensorflow as tf

# Define the model
def make_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Define the secondary model
def make_debiasing_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Define the loss function
def loss_fn(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Define the loss function for the secondary model
def debiasing_loss_fn(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(0.001)

# Define the optimizer for the secondary model
debiasing_optimizer = tf.keras.optimizers.Adam(0.001)

# Define the metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

# Define the metrics for the secondary model
debiasing_train_loss = tf.keras.metrics.Mean(name='debiasing_train_loss')
debiasing_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='debiasing_train_accuracy')

# Define the training step
@tf.function
def train_step(inputs, labels, debiasing_labels):
    with tf.GradientTape() as tape, tf.GradientTape() as debiasing_tape:
        predictions = model(inputs)
        debiasing_predictions = debiasing_model(inputs)

        loss = loss_fn(labels, predictions)
        debiasing_loss = debiasing_loss_fn(debiasing_labels, debiasing_predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    debiasing_gradients = debiasing_tape.gradient(debiasing_loss, debiasing_model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    debiasing_optimizer.apply_gradients(zip(debiasing_gradients, debiasing_model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    debiasing_train_loss(debiasing_loss)
    debiasing_train_accuracy(debiasing_labels, debiasing_predictions)

# Define the training loop
EPOCHS = 5

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        debiasing_labels = tf.random.uniform(shape=(labels.shape[0], 1), minval=0, maxval=2, dtype=tf.int32)
        train_step(images, labels, debiasing_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, De-Biasing Loss: {}, De-Biasing Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          debiasing_train_loss.result(),
                          debiasing_train_accuracy.result()*100))

    train_loss.reset_states()
    train_accuracy.reset_states()
    debiasing_train_loss.reset_states()
    debiasing_train_accuracy.reset_states()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_ds)
print('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_accuracy*100))
