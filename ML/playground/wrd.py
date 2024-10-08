import cv2
import numpy as np
import pickle
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from gtts import gTTS
import os
import time
from pydub import AudioSegment
output_file = "output.mp4"

# Check if the file exists
if os.path.exists(output_file):
    # Remove the file
    os.remove(output_file)
    print(f"Removed '{output_file}'.")
else:
    print(f"The file '{output_file}' does not exist.")

def generate_audio(letter):
    """Generate and save audio for a given letter."""
    tts = gTTS(letter)
    tts.save(f"{letter}.mp3")

# Create a directory for audio files if it doesn't exist
if not os.path.exists('audio_files'):
    os.makedirs('audio_files')
letter_timing = {}
current_letter = ''
start_time = None
# Initialize mediapipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('../model_saves/gesture_15_model')

# Load class names
# with open('../signs/gestures.pkl','rb') as f:
#     actions = pickle.load(f)
# actions = np.array(['Hello', 'Love You', 'Understand', 'Thanks', 'Some', 'Home', 'name', 'my', 'how', 'Sorry', "Help me", "Yes", "No", "eat", "friend"])
actions = np.array([
    "Good morning.",
    "Good evening.",
    "Good night.",
    "How are you.",
    "I am fine.",
    "Eat your food.",
    "I am sorry.",
    "Time is fine.",
    "Give me the book.",
    "I am not happy.",
    "I am coming.",
    "Have a nice day.",
    "I will meet you tomorrow.", 'Sorry', 'Thanks', 'Understand', 'Yes'])
print(actions)

sentence = []
predictions = []

def mediapipe_detection(image, model):
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    results = model.process(image)                 # Make prediction
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

colors = [(245,117,16), (117,245,16), (16,117,245)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*28), (int(prob*100), 90+num*28), colors[1], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# Initialize the webcam
cap = cv2.VideoCapture("uploads\input.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))
while True:

    # Read each frame from the webcam
    _, frame = cap.read()
    if not _ :
        break
    x, y, c = frame.shape
    frame, result = mediapipe_detection(frame, holistic)
    className = ''

    # Process the result
    if result.left_hand_landmarks or result.right_hand_landmarks:
        landmarks = []

        lh = [[res.x, res.y] for res in result.left_hand_landmarks.landmark] if result.left_hand_landmarks else np.zeros(21*2).reshape(-1,2).tolist()
        rh = [[res.x, res.y] for res in result.right_hand_landmarks.landmark] if result.right_hand_landmarks else np.zeros(21*2).reshape(-1,2).tolist()
        
        for i in range(len(lh)):
            landmarks.append([lh[i][0], lh[i][1], rh[i][0], rh[i][1]])
        
        # Drawing landmarks on frames
        mpDraw.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mpDraw.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


        # Predict gesture
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        predictions.append(classID)
        className = actions[classID]
        if className and className != current_letter:
            if current_letter:
                # Calculate the duration for which the previous letter was displayed
                letter_timing[current_letter] = letter_timing.get(current_letter, 0) + (time.time() - start_time)
            
            # Update the current letter and reset the timer
            current_letter = className
            start_time = time.time()

            # Generate audio for the new letter
            audio_file_path = f"audio_files/{className}.mp3"
            if not os.path.exists(audio_file_path):
                generate_audio(className)

        if np.unique(predictions[-20:])[0] == classID: 
                if prediction[0][classID] > 0.5: 
                    
                    if len(sentence) > 0: 
                        if actions[classID] != sentence[-1]:
                            sentence.append(actions[classID])
                    else:
                        sentence.append(actions[classID])

        if len(sentence) > 4: 
            sentence = sentence[-4:]
        

        frame = prob_viz(prediction[0], actions, frame, colors)

    if cv2.waitKey(1) == ord('r'):
            if(len(sentence)!=0):
                sentence.pop()
            # print("Popped: ", sentence)
            
    # show the prediction on the frame
    cv2.rectangle(frame, (0,0), (640, 40), (255, 140, 51), -1)
    cv2.putText(frame, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break
if current_letter:
    letter_timing[current_letter] = letter_timing.get(current_letter, 0) + (time.time() - start_time)

# Combine the audio files based on the timing
combined_audio = None
for letter, duration in letter_timing.items():
    audio = AudioSegment.from_mp3(f"{letter}.mp3")
    repeated_audio = audio * int(duration / audio.duration_seconds)
    combined_audio = combined_audio + repeated_audio if combined_audio else repeated_audio

# Export the combined audio file
combined_audio.export("final_output.mp3", format="mp3")
import subprocess
# release the webcam and destroy all active windows
cap.release()
out.release()
cv2.destroyAllWindows()
subprocess.run(['ffmpeg', '-i', 'output.avi', '-c:v', 'libx264', 'output.mp4'])
os.system("python combiner.py")