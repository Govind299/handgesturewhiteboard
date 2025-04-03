import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

def main():
    st.set_page_config(page_title="Air Canvas", page_icon="✏️", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .big-font {
            font-size:30px !important;
            font-weight: bold;
            color: #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)

    # Title and description
    st.markdown('<p class="big-font">✏️ Air Canvas - Draw in the Air!</p>', unsafe_allow_html=True)
    st.markdown("""
    Draw in real-time using your finger as a brush! 
    - Use your index finger to draw
    - Bring your thumb close to your index finger to stop drawing
    - Use the color buttons to change colors
    """)

    # Initialize session state
    if 'stop_drawing' not in st.session_state:
        st.session_state.stop_drawing = False
    if 'canvas' not in st.session_state:
        st.session_state.canvas = np.zeros((471,636,3)) + 255

    # Two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Camera Feed")
        # Using Streamlit's camera input
        camera_input = st.camera_input("Camera", key="camera")
    
    with col2:
        st.markdown("### Drawing Canvas")
        canvas_placeholder = st.empty()
        
    # Initialize MediaPipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Initialize drawing variables
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]
    blue_index = green_index = red_index = yellow_index = 0
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    if camera_input is not None:
        # Convert the camera input to numpy array
        frame_bytes = camera_input.getvalue()
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Process the frame
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add UI elements
        frame = draw_ui_elements(frame)
        
        # Process hand landmarks
        result = hands.process(framergb)
        
        if result.multi_hand_landmarks:
            landmarks = process_hand_landmarks(result, frame, mpDraw, mpHands)
            if landmarks:
                process_drawing(landmarks, frame, st.session_state.canvas, bpoints, gpoints, 
                              rpoints, ypoints, blue_index, green_index, red_index, 
                              yellow_index, colors, colorIndex)

        # Draw lines
        points = [bpoints, gpoints, rpoints, ypoints]
        draw_lines(points, frame, st.session_state.canvas, colors)

        # Display the processed frame and canvas
        col1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        canvas_placeholder.image(cv2.cvtColor(st.session_state.canvas, cv2.COLOR_BGR2RGB))

    # Clear canvas button
    if st.sidebar.button("Clear Canvas"):
        st.session_state.canvas = np.zeros((471,636,3)) + 255
        bpoints = [deque(maxlen=1024)]
        gpoints = [deque(maxlen=1024)]
        rpoints = [deque(maxlen=1024)]
        ypoints = [deque(maxlen=1024)]
        blue_index = green_index = red_index = yellow_index = 0

def draw_ui_elements(frame):
    # Draw rectangles and text for UI
    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
    frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
    frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)
    
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    return frame

def process_hand_landmarks(result, frame, mpDraw, mpHands):
    landmarks = []
    for handslms in result.multi_hand_landmarks:
        for lm in handslms.landmark:
            lmx = int(lm.x * 640)
            lmy = int(lm.y * 480)
            landmarks.append([lmx, lmy])
        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
    return landmarks

def process_drawing(landmarks, frame, canvas, bpoints, gpoints, rpoints, ypoints,
                   blue_index, green_index, red_index, yellow_index, colors, colorIndex):
    fore_finger = (landmarks[8][0], landmarks[8][1])
    thumb = (landmarks[4][0], landmarks[4][1])
    cv2.circle(frame, fore_finger, 3, (0,255,0), -1)
    
    if (thumb[1]-fore_finger[1]<30):
        bpoints.append(deque(maxlen=512))
        gpoints.append(deque(maxlen=512))
        rpoints.append(deque(maxlen=512))
        ypoints.append(deque(maxlen=512))
        blue_index += 1
        green_index += 1
        red_index += 1
        yellow_index += 1

def draw_lines(points, frame, canvas, colors):
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(canvas, points[i][j][k - 1], points[i][j][k], colors[i], 2)

if __name__ == "__main__":
    main() 
