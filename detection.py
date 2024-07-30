import os
import math
import time
import cv2
import random
import mediapipe as mp
from collections import deque
from cvfpscalc import CvFpsCalc
import pygame

# Initialize Pygame mixer
pygame.mixer.init()

audio_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "audio"
    )
)


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_angle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    angle = math.degrees(math.atan2(y2 - y1, x2 - y1))
    if angle < 0:
        angle += 360
    return angle


def map_value(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


def play_sound(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def main():
    mp_draw = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands
    
    hands = mp_hand.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    
    # Video
    video = cv2.VideoCapture(0)

    # Fullscreen !!!
    # video.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
    # video.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    
    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Get FPS
    cvFpsCalc = CvFpsCalc(buffer_len=20)

    # Dictionary to store middle finger MCP positions for wave detection for each hand
    mcp_positions_dict = {}

    # Gesture names and their corresponding directories
    gesture_names = ["Greetings", "Eyy", "2 Joints", "Rock & Roll", "Peace", "Middle Finger", "Thumbs Up", "Thumbs Down", "Waving"]
    gesture_to_sound = {
        gesture: [os.path.join(audio_path, f"{gesture}", file) 
            for file in os.listdir(os.path.join(audio_path, f"{gesture}")) if file.endswith(".mp3")]
                for gesture in gesture_names
    }

    # 
    audio_on = False
    deltatime = time.time()

    while True:
        fps = cvFpsCalc.get()

        ret, image = video.read()
        image = cv2.flip(image, 1)  # Mirror display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gestures = [] 

        if results.multi_hand_landmarks:
            for idx, (hand_landmark, hand_handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                lmList = []
                for id, lm in enumerate(hand_landmark.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                
                if lmList:
                    # Initialize deque for the current hand if not already present
                    if idx not in mcp_positions_dict:
                        mcp_positions_dict[idx] = deque(maxlen=10)
                    
                    # Determine fingers' status ###################################################
                    thumb_tip = lmList[4]
                    finger_tips = [lmList[8], lmList[12], lmList[16], lmList[20]]
                    wrist = lmList[0]
                    pinky_mcp = lmList[17]
                    middle_mcp = lmList[9]

                    # Determine finger status (open or closed)
                    finger_status = []
                    for _, tip in enumerate(finger_tips):
                        dist = calculate_distance((wrist[1], wrist[2]), (tip[1], tip[2]))
                        # Threshold for other fingers
                        if dist > 60:  # 130
                            finger_status.append(1)  # Finger open
                        else:
                            finger_status.append(0)  # Finger closed

                    # Check the thumb separately
                    thumb_dist = calculate_distance((pinky_mcp[1], pinky_mcp[2]), (thumb_tip[1], thumb_tip[2]))
                    # Threshold for thumb
                    if thumb_dist > 55:  # 110
                        finger_status.insert(0, 1)  # Thumb open
                    else:
                        finger_status.insert(0, 0)  # Thumb closed

                    # print(finger_status)  # !!!

                    # Determine hand's handedness #################################################
                    hand_label = hand_handedness.classification[0].index  # 0 = left, 1 = right

                    # Determine hand's orientation (0 = backhand, 1 = palm) #######################
                    thumb_cmc = lmList[1]
                    thumb_vec = [thumb_cmc[1] - wrist[1], thumb_cmc[2] - wrist[2]]
                    pinky_vec = [pinky_mcp[1] - wrist[1], pinky_mcp[2] - wrist[2]]
                    cross_product = thumb_vec[0] * pinky_vec[1] - thumb_vec[1] * pinky_vec[0]
                    orientation = 0 if (hand_label == 0 and cross_product > 0) or (hand_label == 1 and cross_product < 0) else 1  

                    # Determine hand's angle ######################################################
                    index_mcp = lmList[5]
                    pinky_mcp = lmList[17]
                    index_mcp_coords = (index_mcp[1], index_mcp[2])
                    pinky_mcp_coords = (pinky_mcp[1], pinky_mcp[2])

                    angle = calculate_angle(index_mcp_coords, pinky_mcp_coords)
                    mapped_angle = int(map_value(angle, 0, 360, 0, 12)) * 30

                    # Detect gestures #############################################################
                    gesture_detected = 0  # unrecognized
                    
                    index_middle_distance = calculate_distance(
                        (lmList[8][1], lmList[8][2]), 
                        (lmList[12][1], lmList[12][2]))
                    peacesign_threshold = 25  # 50
                    waving_threshold = 75  # 150

                    if finger_status == [1, 0, 0, 0, 1]:  # eyy
                        gesture_detected = 1
                    elif finger_status == [0, 0, 0, 1, 1]:  # 2 joints
                        gesture_detected = 2

                    elif finger_status[1:5] == [1, 0, 0, 1]:  # rock & roll
                        gesture_detected = 3
                    
                    elif finger_status == [0, 1, 1, 0, 0] and index_middle_distance > peacesign_threshold:  # peace
                        gesture_detected = 4
                    elif finger_status[1:5] == [0, 1, 0, 0] and orientation == 0:  # middle finger
                        gesture_detected = 5

                    elif finger_status == [1, 0, 0, 0, 0]:  # thumbs up & thumbs down
                        if mapped_angle in [30, 60, 90]:
                            gesture_detected = 6
                        elif mapped_angle in [300, 270, 240]:
                            gesture_detected = 7

                    elif finger_status[1:5] == [1, 1, 1, 1] and orientation == 1:  # wave
                        mcp_positions_dict[idx].append(middle_mcp[1])
                        if len(mcp_positions_dict[idx]) == mcp_positions_dict[idx].maxlen:
                            diffs = [abs(mcp_positions_dict[idx][i] - mcp_positions_dict[idx][i + 1]) for i in range(len(mcp_positions_dict[idx]) - 1)]
                            if sum(diffs) > waving_threshold:
                                gesture_detected = 8

                    gestures.append(gesture_detected)
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
            print(gestures)  # !!!

            # 1 eyy
            # 2 2joints
            # 3 rock&roll
            # 4 peace
            # 5 middle finger
            # 6 thumbs up
            # 7 thumbs down
            # 8 waving

            # Play Sound ##########################################################################
            currenttime = time.time()
            sound_files = gesture_to_sound[gesture_names[gesture_detected]]
            if len(sound_files) > 0:  # !!!
                if not audio_on:
                    sound_file = random.choice(sound_files)
                    play_sound(sound_file)
                    audio_on = True
                    deltatime = currenttime
                else:
                    if not pygame.mixer.music.get_busy() and (currenttime - deltatime) >= 3:
                        audio_on = False
                        deltatime = currenttime

        cv2.putText(image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 100, 100), 2, cv2.LINE_AA)

        cv2.imshow("Frame", image)
        key = cv2.waitKey(1)
        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
