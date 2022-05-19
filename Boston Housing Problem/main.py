import tensorflow as tf
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler

# get the data
data = tf.keras.datasets.boston_housing.load_data(path='boston_housing.npz', test_split=0.2, seed=113)

# Split the data
train_data = data[0]
test_data = data[1]
X_train = pd.DataFrame(train_data[0],
                       columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                                'LSTAT'])
Y_train = pd.DataFrame(train_data[1])
X_test = pd.DataFrame(test_data[0],
                      columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                               'LSTAT'])
Y_test = pd.DataFrame(test_data[1])

# Normalize the Data
ct = make_column_transformer(
    (MinMaxScaler(), ['CRIM', 'ZN', 'INDUS', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']))

ct.fit(X_train)

X_train_norm = ct.transform(X_train)
X_test_norm = ct.transform(X_test)

# set seed
tf.random.set_seed(42)
# Create the model
model = tf.keras.Sequential(name="BostonHouseModelNormData")
model.add(tf.keras.layers.Dense(100))
model.add(tf.keras.layers.Dense(50))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["mae"])

# Fit The model
model.fit(X_train_norm, Y_train, epochs=200, verbose=1)

# get Summary of Model
print(model.summary())

# Evaluate teh Model
model.evaluate(X_test_norm, Y_test)
