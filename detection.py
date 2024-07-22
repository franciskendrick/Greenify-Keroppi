import cv2
import math
import mediapipe as mp
from collections import deque
from cvfpscalc import CvFpsCalc


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_angle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 360
    return angle


def map_value(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


def main():
    mp_draw = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands
    
    hands = mp_hand.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    
    video = cv2.VideoCapture(0)
    cvFpsCalc = CvFpsCalc(buffer_len=20)

    # Queue to store middle finger MCP positions for wave detection
    mcp_positions = deque(maxlen=10)  # Adjust the length as needed
    
    while True:
        fps = cvFpsCalc.get()

        ret, image = video.read()
        image = cv2.flip(image, 1)  # Mirror display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmark, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lmList = []
                for id, lm in enumerate(hand_landmark.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                
                if lmList:
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
                        if dist > 130:
                            finger_status.append(1)  # Finger open
                        else:
                            finger_status.append(0)  # Finger closed

                    # Check the thumb separately
                    thumb_dist = calculate_distance((pinky_mcp[1], pinky_mcp[2]), (thumb_tip[1], thumb_tip[2]))
                    # Threshold for thumb
                    if thumb_dist > 110:
                        finger_status.insert(0, 1)  # Thumb open
                    else:
                        finger_status.insert(0, 0)  # Thumb closed

                    # Determine hand's handedness #################################################
                    hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'

                    # Determine hand's orientation (0 = backhand, 1 = palm) #######################
                    thumb_cmc = lmList[1]
                    thumb_vec = [thumb_cmc[1] - wrist[1], thumb_cmc[2] - wrist[2]]
                    pinky_vec = [pinky_mcp[1] - wrist[1], pinky_mcp[2] - wrist[2]]
                    cross_product = thumb_vec[0] * pinky_vec[1] - thumb_vec[1] * pinky_vec[0]
                    orientation = 0 if (hand_label == 'Left' and cross_product > 0) or (hand_label == 'Right' and cross_product < 0) else 1  

                    # Determine hand's angle ######################################################
                    index_mcp = lmList[5]
                    pinky_mcp = lmList[17]
                    index_mcp_coords = (index_mcp[1], index_mcp[2])
                    pinky_mcp_coords = (pinky_mcp[1], pinky_mcp[2])

                    angle = calculate_angle(index_mcp_coords, pinky_mcp_coords)
                    mapped_angle = int(map_value(angle, 0, 360, 0, 12)) * 30

                    # Detect gestures #############################################################
                    gesture_detected = "unrecognized"
                    
                    index_middle_distance = calculate_distance(
                        (lmList[8][1], lmList[8][2]), 
                        (lmList[12][1], lmList[12][2]))
                    peacesign_threshold = 50
                    waving_threshold = 150

                    if finger_status == [1, 0, 0, 0, 1]:  # eyy
                        gesture_detected = "eyy"
                    elif finger_status == [0, 0, 0, 1, 1]:  # 2 joints
                        gesture_detected = "2joints"

                    elif finger_status[1:5] == [1, 0, 0, 1]:  # rock & roll
                        gesture_detected = "rock&roll"
                    
                    elif finger_status == [0, 1, 1, 0, 0] and index_middle_distance > peacesign_threshold:  # peace
                        gesture_detected = "peace"
                    elif finger_status[1:5] == [0, 1, 0, 0] and orientation == 0:  # middle finger
                        gesture_detected = "middle"

                    elif finger_status == [1, 0, 0, 0, 0]:  # thumbs up & thumbs down
                        if mapped_angle in [30, 60, 90]:
                            gesture_detected = "thumbs up"
                        elif mapped_angle in [300, 270, 240]:
                            gesture_detected = "thumbs down"

                    elif finger_status[1:5] == [1, 1, 1, 1] and orientation == 1:  # wave
                        mcp_positions.append(middle_mcp[1])
                        if len(mcp_positions) == mcp_positions.maxlen:
                            diffs = [abs(mcp_positions[i] - mcp_positions[i + 1]) for i in range(len(mcp_positions) - 1)]
                            if sum(diffs) > waving_threshold:
                                gesture_detected = "waving"

                    print(gesture_detected)

                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)        

        cv2.putText(image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 100, 100), 2, cv2.LINE_AA)

        cv2.imshow("Frame", image)
        key = cv2.waitKey(1)
        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
