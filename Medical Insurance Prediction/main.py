# Final Model
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# Create a column Transformer
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),  # turn all values between 0-1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Create X and Y values again
X = data.drop("charges", axis=1)
y = data["charges"]

# Build our train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the column transformer to our training data
ct.fit(X_train)

# #Transform training and testing data with normalization (MinMaxScaler) and OneHotEncoder
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# set seed
tf.random.set_seed(42)
# Create a model as per above specifications
insurance_model = tf.keras.Sequential(name="InsuranceModel")
insurance_model.add(tf.keras.layers.Dense(200))
insurance_model.add(tf.keras.layers.Dense(100))
insurance_model.add(tf.keras.layers.Dense(50))
insurance_model.add(tf.keras.layers.Dense(1))

# Compile the model
insurance_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["mae"])

# Fit the model
insurance_model.fit(X_train_normal, y_train, epochs=300, verbose=0)

# Evaluate the Model
insurance_model.evaluate(X_test_normal, y_test)
