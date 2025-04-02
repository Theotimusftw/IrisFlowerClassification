import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
print(train)

plt.scatter(
    train["sepal_length"],
    train["sepal_width"],
    c=train["species"]
)
plt.show()
plt.scatter(
    train["petal_length"],
    train["petal_width"],
    c=train["species"]
)
plt.show()

petal_ratio = list(zip(train["petal_length"], train["petal_width"]))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(petal_ratio, train["species"])

validate.drop(columns="species")
test.drop(columns="species")

