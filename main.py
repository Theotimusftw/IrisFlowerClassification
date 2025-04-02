import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder 
from sklearn.neighbors import KNeighborsClassifier


df = pd.DataFrame(pd.read_csv("~/Documents/PythonProjects/IrisFlowerClassification/IRIS.csv"))
le = LabelEncoder()
le.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_arr, validate_arr, test_arr = np.split(
        df.sample(frac=1, random_state=42).to_numpy(),  # Convert to NumPy array before splitting
        [int(0.6 * len(df)), int(0.8 * len(df))]
    )
    
    # Convert NumPy arrays back to DataFrames
    train = pd.DataFrame(train_arr, columns=df.columns)
    validate = pd.DataFrame(validate_arr, columns=df.columns)
    test = pd.DataFrame(test_arr, columns=df.columns) 
    return train, validate, test


train, validate, test = split_dataset(df)
train["species"] = le.transform(train["species"])
test["species"] = le.transform(test["species"])
validate_check = validate
test_check = test
validate.drop(columns="species")
test.drop(columns="species")


def scatterplot(df: DataFrame, x: str, y: str, c: str) -> None:
    plt.scatter(x=df[x], y=df[y], c=df[c])
    plt.show()


scatterplot(train, "sepal_length", "sepal_width", "species")
scatterplot(train, "petal_length", "petal_width", "species")


petal_ratio = list(zip(train["petal_length"], train["petal_width"]))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(petal_ratio, train["species"])
prediction = knn.predict(list(zip(validate["petal_length"], validate["petal_width"])))


plt.scatter(validate["petal_length"], validate["petal_width"], c=prediction)
plt.text(
    x=validate["petal_length"]-1.7,
    y=validate["petal_width"]-0.7,
    s=f"something"
)
plt.show()
