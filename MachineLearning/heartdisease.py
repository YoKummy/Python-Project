import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

#load dataset
dataset = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
x = dataset.drop(columns=["target"])
y = dataset["target"]

#split data into training, validating, testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#normalize the input features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

#build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#compile model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)

#train the model with early stopping
history = model.fit(x_train_scaled, y_train, validation_data = (x_val_scaled, y_val), epochs = 1000, callbacks = [early_stopping, model_checkpoint])

#evaluate the best model on the test state
best_model = tf.keras.models.load_model("best_model.keras")
test_loss, test_accuracy = best_model.evaluate(x_test_scaled, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

""" score = model.evaluate(x_test, y_test, verbose=0)
print("Loss: ", score[0])
print("Accuracy: ", score[1]) test"""