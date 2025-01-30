import itertools
import numpy as np
import cv2 as cv
import mediapipe as mp
from fastapi import FastAPI, WebSocket
import base64
from model import KeyPointClassifier
from model import PointHistoryClassifier
from collections import deque, Counter
import copy
import uvicorn

app = FastAPI()


class GestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Initialize classifiers
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        # Load gesture labels
        self.keypoint_classifier_labels = self._load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')

        # Initialize history
        self.history_length = 16
        self.point_histories = {}  # Dict to store histories for each participant
        self.finger_gesture_histories = {}  # Dict to store gesture histories

    def _load_labels(self, path):
        import csv
        with open(path, encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
        return keypoint_classifier_labels

    def _process_frame(self, frame, participant_id):
        # Initialize histories for new participant
        if participant_id not in self.point_histories:
            self.point_histories[participant_id] = deque(maxlen=self.history_length)
            self.finger_gesture_histories[participant_id] = deque(maxlen=self.history_length)

        # Convert frame to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is None:
            return None

        # Process each detected hand
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Calculate landmarks
            landmark_list = self._calc_landmark_list(frame, hand_landmarks)
            processed_landmark_list = self._pre_process_landmark(landmark_list)

            # Classify hand sign
            hand_sign_id = self.keypoint_classifier(processed_landmark_list)
            if hand_sign_id >= 0 and hand_sign_id < len(self.keypoint_classifier_labels):
                return f"{participant_id}: {self.keypoint_classifier_labels[hand_sign_id]}"

        return None

    def _calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def _pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list


# Initialize gesture recognizer
gesture_recognizer = GestureRecognizer()


@app.websocket("/ws/gesture")
async def gesture_recognition(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            participant_id = data['participant_id']
            frame_data = base64.b64decode(data['frame'])

            # Convert frame data to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

            # Process frame
            result = gesture_recognizer._process_frame(frame, participant_id)

            # Send result if gesture detected
            if result:
                await websocket.send_json({
                    'status': 'success',
                    'gesture': result
                })

    except Exception as e:
        await websocket.send_json({
            'status': 'error',
            'message': str(e)
        })

    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)