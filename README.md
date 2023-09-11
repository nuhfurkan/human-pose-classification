# Human Pose Classification
A human pose classification ML Modal and web interface with a server.

Note:
You should add mediapipe heavytas to the "flask-ront-end/serverfiles" with the following name "pose_landmarker_heavy.task".
You can find the heavy task in the link below.
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index

## Functions Used in the Project
### Data Processing
1. normalize_dataset
   Params:
   - dataSubSet -> pandas.DataFrame
   Return:
   - pandas.DataFrame
   This function receives a pandas.DataFrame and return a processed pandas.DataFrame. Function normalises the data by dividing all the values into the maximum entry in the parameter DataFrame. It also substracts the minimum value from all the entries in the DataFrame. So all the values would be reframed into values between 0 to 1.
3. normalize_dataframe
   Params:
   - dataAsDF -> pandas.DataFrame (default: False)
   - extract -> Boolean
   Return:
   - pandas.DataFrame
   This function send several calls to the normalize_dataset function to normalize X, Y and Z correspondingly. If extract is "True", the return values extracted to a file named as "normalized_landmarks.csv".
   
5. train_modal
6. train_knn_modal
7. pickle_modal

### mpprocess
mpprocess file contains the class of MPObject. MPObject class has the following methods:
1. __init__
2. runMLPC
3. fetchResults
4. draw_landmarks_on_image
5. retrieveLandmarks
6. putDataInFrame
7. normalize_dataset
8. normalizeData
9. 
