import mediapipe as mp
import pandas as pd
from joblib import load
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from matplotlib import pyplot as plt
            
columnNames = [
        "x_nose",
        "y_nose",
        "z_nose",
        "x_left_eye_inner",
        "y_left_eye_inner",
        "z_left_eye_inner",
        "x_left_eye",
        "y_left_eye",
        "z_left_eye",
        "x_left_eye_outer",
        "y_left_eye_outer",
        "z_left_eye_outer",
        "x_right_eye_inner",
        "y_right_eye_inner",
        "z_right_eye_inner",
        "x_right_eye",
        "y_right_eye",
        "z_right_eye",
        "x_right_eye_outer",
        "y_right_eye_outer",
        "z_right_eye_outer",
        "x_left_ear",
        "y_left_ear",
        "z_left_ear",
        "x_right_ear",
        "y_right_ear",
        "z_right_ear",
        "x_mouth_left",
        "y_mouth_left",
        "z_mouth_left",
        "x_mouth_right",
        "y_mouth_right",
        "z_mouth_right",
        "x_left_shoulder",
        "y_left_shoulder",
        "z_left_shoulder",
        "x_right_shoulder",
        "y_right_shoulder",
        "z_right_shoulder",
        "x_left_elbow",
        "y_left_elbow",
        "z_left_elbow",
        "x_right_elbow",
        "y_right_elbow",
        "z_right_elbow",
        "x_left_wrist",
        "y_left_wrist",
        "z_left_wrist",
        "x_right_wrist",
        "y_right_wrist",
        "z_right_wrist",
        "x_left_pinky_1",
        "y_left_pinky_1",
        "z_left_pinky_1",
        "x_right_pinky_1",
        "y_right_pinky_1",
        "z_right_pinky_1",
        "x_left_index_1",
        "y_left_index_1",
        "z_left_index_1",
        "x_right_index_1",
        "y_right_index_1",
        "z_right_index_1",
        "x_left_thumb_2",
        "y_left_thumb_2",
        "z_left_thumb_2",
        "x_right_thumb_2",
        "y_right_thumb_2",
        "z_right_thumb_2",
        "x_left_hip",
        "y_left_hip",
        "z_left_hip",
        "x_right_hip",
        "y_right_hip",
        "z_right_hip",
        "x_left_knee",
        "y_left_knee",
        "z_left_knee",
        "x_right_knee",
        "y_right_knee",
        "z_right_knee",
        "x_left_ankle",
        "y_left_ankle",
        "z_left_ankle",
        "x_right_ankle",
        "y_right_ankle",
        "z_right_ankle",
        "x_left_heel",
        "y_left_heel",
        "z_left_heel",
        "x_right_heel",
        "y_right_heel",
        "z_right_heel",
        "x_left_foot_index",
        "y_left_foot_index",
        "z_left_foot_index",
        "x_right_foot_index",
        "y_right_foot_index",
        "z_right_foot_index"
    ]

class MPObject:
    def __init__(self) -> None:
        model_path = "serverfiles/pose_landmarker_heavy.task"
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.IMAGE
        )
        self.mlpclassifier = load("serverfiles/model.joblib")
        pass

    def runMLPC(self, data, prt=False):
        predictions = self.mlpclassifier.predict(data)
        print(predictions)
        print("Prediction Done")
        return predictions

    def fetchResults(self, imagelocation):
        return self.runMLPC(
                self.normalizeData(
                    self.putDataInFrame(
                        self.retrieveLandmarks(imagelocation=imagelocation, prt=True)
                    )
                )
        )

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
            print("tpye of the image\n")
            print(type(annotated_image))
            plt.imshow(annotated_image, interpolation='nearest')
            plt.show()

    def retrieveLandmarks(self, imagelocation, prt=False):
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            mp_image = mp.Image.create_from_file(imagelocation)
            #mp_image = mp.Image.create_from_file("static/files/testimg.jpeg")
            pose_landmarker_result = landmarker.detect(mp_image)
            if prt:
                print(pose_landmarker_result.pose_world_landmarks)
            
            if len(pose_landmarker_result.pose_world_landmarks) == 0:
                return

            self.draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)

            return pose_landmarker_result.pose_world_landmarks
        
    def putDataInFrame(self, poses) -> pd.DataFrame:
        values = []
        for pose in poses[0]:
            values.append(pose.x)
            values.append(pose.y)
            values.append(pose.z)

        return pd.DataFrame(data=[values], columns=columnNames)
    
        
    def normalize_dataset(self, dataList: pd.DataFrame) -> pd.DataFrame:
        toProces = dataList.iloc[0]
        minValue = toProces.min()

        # Make all the values in the row positive and set minimum value as zero
        toProces -= minValue

        maxValue = toProces.max()
        normalized_row = toProces/maxValue
        dataList.iloc[0] = normalized_row

        return dataList

    # normalization of the points should be seperated for x, y and, z values
    # Normalize points here
    def normalizeData(self, dataAsDF:pd.DataFrame, prt = False) -> pd.DataFrame:
        xValues = dataAsDF[dataAsDF.columns[dataAsDF.columns.str.startswith("x")]]
        xNormalized = self.normalize_dataset(xValues)

        yValues = dataAsDF[dataAsDF.columns[dataAsDF.columns.str.startswith("y")]]
        yNormalized = self.normalize_dataset(yValues)
    
        zValues = dataAsDF[dataAsDF.columns[dataAsDF.columns.str.startswith("z")]]
        zNormalised = self.normalize_dataset(zValues)

        # print ("checkpoint")
        normalised_landmarks = pd.concat([xNormalized, yNormalized], axis=1)
        normalised_landmarks = pd.concat([normalised_landmarks, zNormalised], axis=1)

        if prt == True:
            normalised_landmarks.to_csv("normalized_landmarks.csv")
            print(normalised_landmarks.head())

        return normalised_landmarks