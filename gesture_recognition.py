import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
tip_ids = [4, 8, 12, 16, 20]

def get_gesture(f):
    if f == [0,0,0,0,0]: return "ROCK ✊"
    if f == [1,1,1,1,1]: return "PAPER(5) ✋"
    if f == [0,1,1,0,0]: return "SCISSORS(2)✌"
    if f == [1,0,0,0,0]: return "LIKE 👍"
    if f == [1,0,0,0,1]: return "CALL 🤙"
    if f == [0,1,0,0,0]: return "ONE ☝"
    if f == [0,1,1,1,0]: return "THREE 🤟"
    if f == [1,1,0,0,1]: return "I LOVE YOU ❤️"
    if f == [1,1,0,0,0]: return "OK 👌"
    if f == [0,1,1,1,1]: return "FOUR ✋"
    return "UNKNOWN"


def fingers_up(lmList):
    fingers = []

    fingers.append(lmList[4][0] > lmList[3][0])
    for i in [8,12,16,20]:
        fingers.append(lmList[i][1] < lmList[i-2][1])
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        lmList = []
        hand = results.multi_hand_landmarks[0]

        for id, lm in enumerate(hand.landmark):
            h, w, _ = img.shape
            lmList.append((int(lm.x*w), int(lm.y*h)))

        fingers = fingers_up(lmList)
        gesture = get_gesture(fingers)

        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        cv2.rectangle(img, (20,20), (350,120), (0,255,0), -1)
        cv2.putText(img, f'Gesture: {gesture}', (40,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,0), 4)

    cv2.imshow("AI Hand Gesture Shape Recognition", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
