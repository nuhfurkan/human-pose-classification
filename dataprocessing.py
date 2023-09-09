from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump

pd.options.mode.chained_assignment = None

df = pd.read_csv("landmarks.csv")
labels = pd.read_csv("labels.csv")

# this line added after running normalize_dataframe function
normalized_landmarks = pd.read_csv("normalized_landmarks.csv")

def normalize_dataset(dataSubSet: pd.DataFrame) -> pd.DataFrame:
    for index, row in dataSubSet.iterrows():
        toProces = row[:-1]
        minValue = toProces.min()

        # Make all the values in the row positive and set minimum value as zero
        toProces -= minValue

        maxValue = toProces.max()
        normalized_row = toProces/maxValue

        # print(normalized_row)
        dataSubSet.loc[index, dataSubSet.columns[:-1]] = normalized_row
    
    return dataSubSet

# normalization of the points should be seperated for x, y and, z values
# Normalize points here
def normalize_dataframe(dataAsDF:pd.DataFrame, extract = False) -> pd.DataFrame:
    xValues = dataAsDF[dataAsDF.columns[dataAsDF.columns.str.startswith("x")].append(
            (dataAsDF.columns[dataAsDF.columns.str.startswith("pose_id")])
        )]
    xNormalized = normalize_dataset(xValues)

    yValues = dataAsDF[dataAsDF.columns[dataAsDF.columns.str.startswith("y")].append(
            (dataAsDF.columns[dataAsDF.columns.str.startswith("pose_id")])
        )]
    yNormalized = normalize_dataset(yValues)
    
    zValues = dataAsDF[dataAsDF.columns[dataAsDF.columns.str.startswith("z")].append(
            (dataAsDF.columns[dataAsDF.columns.str.startswith("pose_id")])
        )]
    zNormalised = normalize_dataset(zValues)

    normalised_landmarks = pd.merge(pd.merge(xNormalized, yNormalized, on='pose_id'), zNormalised, on='pose_id')
    
    if extract == True:
        normalised_landmarks.to_csv("normalized_landmarks.csv")

    return normalised_landmarks

def train_modal(processed_df:pd.DataFrame) -> MLPClassifier:
    X_train, X_test, y_train, y_test = train_test_split(processed_df, labels["pose"], test_size=0.07)
    modal = MLPClassifier()
    # For solver lbfgs and adam results are similar close to each other
    # For solver sgd results accuracy is really low
    modal = MLPClassifier(max_iter=500, solver="lbfgs", hidden_layer_sizes=(80))
    modal.fit(X_train, y_train)
    predictions = modal.predict(X_test)
    print(
        accuracy_score(y_test, predictions),
        "\n",
        confusion_matrix(y_test, predictions)
    )

    cm = confusion_matrix(y_test, predictions, labels=modal.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=modal.classes_)
    disp.plot()
    plt.savefig('confusion_matrix.png')
    plt.show()

    return modal

def train_knn_modal(processed_df:pd.DataFrame) -> KNeighborsClassifier:
    X_train, X_test, y_train, y_test = train_test_split(processed_df, labels["pose"], test_size=0.1)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    predictions = neigh.predict(X_test)

    # I did not believe that it performed near perfect so that I checked the results manually
    merged_knn_data = pd.DataFrame({
        'y_test': y_test,
        'predictions': predictions
    }).to_csv("knn_accuracy.csv")
    print(merged_knn_data)

    print(
        accuracy_score(y_test, predictions)
    )
    return neigh

def pickle_modal(modal_to_pickle):
    pickled_modal = dump(modal_to_pickle, "model.joblib")
    pass

# data was normalised and extracted to the file
# normalized_landmarks.csv
### CALL NORMALIZE_DATAFRAME HERE ###
# normalize_dataframe(df)

# Out of curiosity I trained a knn model too
# train_knn_modal(normalized_landmarks)

### Train modal and pickle it for later use ###
normalized_landmarks = normalized_landmarks.drop(columns=["pose_id", "Unnamed: 0"])
modal_trained = train_modal(normalized_landmarks)
pickle_modal(modal_trained)

"""
Readings of the pickled modal

Accuracy:
0.8762886597938144 

Confusion Matrix:
 [[16  0  0  0  1  0  0  0  1  0]
 [ 0  7  1  0  0  0  0  0  0  0]
 [ 0  0 16  0  0  0  0  1  0  0]
 [ 0  0  0  5  0  0  0  1  0  2]
 [ 0  0  0  0  5  1  0  0  0  0]
 [ 0  0  0  0  1 10  1  0  0  0]
 [ 0  0  0  0  0  0  8  0  0  0]
 [ 0  0  0  0  0  0  0  6  0  0]
 [ 0  0  0  0  0  0  0  1  7  0]
 [ 1  0  0  0  0  0  0  0  0  5]]

"""