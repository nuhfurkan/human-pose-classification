# Human Pose Classification
A human pose classification ML Modal and web interface with a server.

## Installing dependencies
In order to install dependencies first create your own virtual environment using following line of code
> python3 -m venv #YOUR-ENV-NAME#

Then activate your virtual environment

Then type following command to install dependencies
> python3 -m pip install -r requirements.txt

<br>

Note:
You should add mediapipe heavytask to the "flask-front-end/serverfiles" with the following name "pose_landmarker_heavy.task".
You can find the heavy task in the link below.
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index

<br>

## Functions Used in the Project
### Data Processing
1. normalize_dataset

   Params:
   
   > dataSubSet -> pandas.DataFrame
   
   Return:
   
   > pandas.DataFrame
   
   This function receives a pandas.DataFrame and return a processed pandas.DataFrame. Function normalises the data by dividing all the values into the maximum entry in the parameter DataFrame. It also substracts the minimum value from all the entries in the DataFrame. So all the values would be reframed into values between 0 to 1.

2. normalize_dataframe

   Params:

   > dataAsDF -> pandas.DataFrame (default: False)
   >
   > extract -> Boolean
   
   Return:

   > pandas.DataFrame
   
   This function send several calls to the normalize_dataset function to normalize X, Y and Z correspondingly. If extract is "True", the return values extracted to a file named as "normalized_landmarks.csv".
   
3. train_modal

   Params

   > processed_df -> pandas.DataFrame

   Return

   > sklearn.neural_network.MLPClassifier

   This function trains a MLPClassifier modal with following parameters (max_ites: 500, solver: "lbfgs", hidden_layer_size: 80).
   Function also saves the confusion matrix to a file "confusion_matrix.png".
   The test train split of data 0.07.

4. train_knn_modal

   Params:

   > processed_df -> pandas.DataFrame
   
   Return:

   > sklearn.neighbors.KNeighborsClassifier
   
   This function trains a KNeighborsClassifier modal with three neighbours.
   Later, function prints the accuracy score and export the results of predictions to file "knn_accuracy.csv".
   The test train split of data 0.1.

5. pickle_modal

   Params:

   > modal_to_pickle -> MLPClassifier
   
   Return:

   > NaN
   
   Functions pickles the modal using joblib to a file named "modal.joblib".

### mpprocess
mpprocess file contains the class of MPObject. MPObject class has the following methods:
1. \__init__
   Params:

   > Nan

   Return:

   > Nan

   Method initialises the MPObject objects. It sets the MediaPipe pose_landmarker_heavy.task options and loads the pickled MPClassifier modal.

2. runMLPC

   Params:

   > data -> pandas.DataFrame
   >
   > prt -> Boolean (default: False)

   Return:

   > prediction -> ndarray

   This method returns the prediction class for a single entry of data. ptr is depreciated.

3. fetchResults

   Params:

   > imagelocation -> str
   
   Return:

   > prediction -> ndarray
   
   This method runs all the necessary functions to classify a pose from an image whose location stated in the parameter, "iamgelocation".

4. draw_landmarks_on_image

   Params:
   
   > rgb_image -> numpy.ndarray.view
   >
   > detection_results -> poseLandmarkerResult

   Return:

   > NaN

   This medhod draws the landmarks to the image and shows. It gets the image and landmarks.

5. retrieveLandmarks

   Params:

   > imagelocation -> str
   >
   > prt -> Boolean (default: False)

   Return:

   > poseLandmarkerResult.pose_world_landmarks

   This methods calls the relevant MediaPipe functions to extract the landmarks and returns the landmarks.

6. putDataInFrame

   Params:

   > poses -> poseLandmarkerResult.pose_world_landmarks
   
   Return:

   > pandas.DataFrame

   This methods receives a MediaPipe pose_world_landmarks object and converts it to pandas.DataFrame.


7. normalize_dataset

   Params:

   > dataList -> pandas.DataFrame
   
   Return:

   > pandas.DataFrame

   This method normalizes the data entered by dividing it to the maximum entry value and also substracts the minimum value from all entries.

8. normalizeData

   Params:

   > dataAsDF -> pandas.DataFrame
   >
   > prt -> Boolean (default: False) 
   
   Return:

   > pandas.DataFrame

   This method normalizes the data read through MediaPipe functions and other preprocessing functions. If prt is "True", normalized landmarks will be extracted to the file "normalized_landmarks.csv".

## Notes

More data can be found in the files given above in the "NNGA Project_ A Pose Classifier.pdf".

Also some relevant links to the project can be found in the "notes.txt" file.

