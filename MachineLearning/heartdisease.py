import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
dataset = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
x = dataset.drop(columns=["target"])
y = dataset["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation = "sigmoid"))
model.add(tf.keras.layers.Dense(256, activation = "sigmoid"))
model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["Accuracy"])

model.fit(x_train, y_train, epochs = 1000)

score = model.evaluate(x_test, y_test, verbose=0)
print("Loss: ", score[0])
print("Accuracy: ", score[1])