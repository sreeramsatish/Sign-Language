import streamlit as st
import os
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def save_uploaded_file(uploaded_file, base_name):
    try:
        with open(os.path.join('uploads', base_name + os.path.splitext(uploaded_file.name)[1]), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False
# Set page title and background image
st.set_page_config(page_title="Hand Sign Recognition", page_icon=":guardsman:", layout="wide")
background_image = "https://images.unsplash.com/photo-1599892317428-8f59137fa055?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8aGFuZCUyMHNpZ25zJTIwcGhvdG8lMjBhdCUyMGltYWdlfGVufDB8fDB8fA%3D%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60"
page_bg_img = '''
<style>
body {
background-image: url("%s");
background-size: cover;
}
</style>
''' % background_image

# Set up sidebar and main content area
options = ["Online Hand Posture Acknowledgement for Deaf and Dumb People","uploader"]
st.sidebar.title("Online Hand Posture Acknowledgement for Deaf and Dumb People")
selected_language = st.sidebar.radio("", options)
st.sidebar.markdown("---")
button1 = st.sidebar.button("important words")
button2 = st.sidebar.button("Words")

st.markdown(page_bg_img, unsafe_allow_html=True)

# Display content based on language selection and button clicks
if selected_language == "Online Hand Posture Acknowledgement for Deaf and Dumb People":
    st.header("Hand Posture Acknowledgement App")
    st.image("https://media.tenor.com/RID3Av9YUA8AAAAC/hand-signs-famous.gif")
    st.write('Sign language is a visual language that uses a combination of hand gestures, facial expressions, and body language to communicate. It is a unique language with its own grammar and syntax, and is used by millions of people around the world who are deaf or hard of hearing. Sign languages vary across different regions and countries, and include Online Hand Posture Acknowledgement for Deaf and Dumb People (ASL), British Sign Language (BSL), Auslan (Australian Sign Language), and many more. Sign language is not simply a translation of spoken language, but is a rich and complex language in its own right. It allows individuals who are deaf or hard of hearing to communicate with others and fully participate in their communities, and has been recognized as a national language in many countries. With the help of technology, sign language is becoming more accessible than ever before, with tools such as video relay services and sign language interpreting services enabling people to communicate more easily and effectively.')
    if button1:
        st.subheader("Alphabets recognition")
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/88/Greek--alphabet-%28upper-case%29-animated.gif")
        import cv2
        import numpy as np
        import math
        import pickle
        import mediapipe as mp
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from gtts import gTTS
        import os
        import time
        from pydub import AudioSegment
        import os
        os.remove("output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

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

        # initialize mediapipe
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils

        # Load the gesture recognizer model
        model = load_model('../model_saves//checkpoints/alphabets_Z_model')

        # Load class names
        with open("../signs/alphabets.pkl", 'rb') as f:
            actions = pickle.load(f)
        actions = np.array(["I'm working on a new project", "I'm not free", 'I really appreciate it', "Let's grab a cup of coffee","I'm looking for a new job", 'Had your food?', 'I need', 'Today', 'tomorrow', 'yesterday', 'bye', 'go','good', 'morning', 'evening', 'night', 'hello', 'please', 'help', 'money', "it's", 'fine', 'okay', 'cool', 'meet', "let's"])

        print(actions)

        sentence = []
        predictions = []

        def mediapipe_detection(image, model):
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
            results = model.process(image)                 # Make prediction
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
            return image, results


        def getAngle(a, b, c):
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang


        colors = [(245,117,16), (117,245,16), (16,117,245)]

        def prob_viz(res, actions, input_frame, colors):
            output_frame = input_frame.copy()
            for num, prob in enumerate(res):
                cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[1], -1)
                cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                
            return output_frame


        # Initialize the webcam
        cap = cv2.VideoCapture(0)

        while True:
            

            # Read each frame from the webcam
            _, frame = cap.read()
            x, y, c = frame.shape

            frame, result = mediapipe_detection(frame, hands)
            className = ''
            
            # post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:

                    angle_1 = getAngle((handslms.landmark[mpHands.HandLandmark.THUMB_TIP].x, handslms.landmark[mpHands.HandLandmark.THUMB_TIP].y),
                                                    (handslms.landmark[mpHands.HandLandmark.WRIST].x, handslms.landmark[mpHands.HandLandmark.WRIST].y),
                                                    (handslms.landmark[mpHands.HandLandmark.PINKY_TIP].x, handslms.landmark[mpHands.HandLandmark.PINKY_TIP].y))

                    angle_2 = getAngle((handslms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x, handslms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y),
                                                (handslms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].x, handslms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].y),
                                                (handslms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x, handslms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y))
                    
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)

                        landmarks.append([lmx, lmy])

                    # landmarks.append([angle_1, angle_2])
                    
                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

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
                    else:
                        print("no")
                    if np.unique(predictions[-20:])[0] == classID: 
                        if prediction[0][classID] > 0.7: 
                            
                            if len(sentence) > 0: 
                                if actions[classID] != sentence[-1]:
                                    sentence.append(actions[classID])
                            else:
                                sentence.append(actions[classID])

                    if len(sentence) > 15: 
                        sentence = sentence[-15:]
                    
                    frame = prob_viz(prediction[0], actions, frame, colors)
            
            if cv2.waitKey(1) == ord('r'):
                    if len(sentence) > 0:
                        sentence.pop()

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
    elif button2:
        st.subheader("Words recognition")
        st.image("https://media.tenor.com/mrWq7zzn4ugAAAAM/words-text.gif")
                # Import necessary packages
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
        os.remove("output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
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
        actions = np.array(['eat', 'friend', 'Hello', 'Help me', 'Home', 'how', 'Love You', 'my', 'name', 'No', 'Some', 'Sorry', 'Thanks', 'Understand', 'Yes'])
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
        cap = cv2.VideoCapture(0)

        while True:

            # Read each frame from the webcam
            _, frame = cap.read()
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
                        if prediction[0][classID] > 0.7: 
                            
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
if selected_language == "uploader":
    st.title('Upload the File')

    # File uploader and text input for base name
    uploaded_file = st.file_uploader("Choose a file")
    base_name = "input"

    if uploaded_file is not None and base_name:
        # Save the file
        if save_uploaded_file(uploaded_file, base_name):
            st.success('File saved successfully.')
        else:
            st.error('Error saving file.')

    # Buttons to print 'Hi' and 'Bye'
    if st.button('words'):
        os.system("python alpha.py")
        st.video("supream.mp4")
    
    if st.button('base words'):
        os.system("python wrd.py")
        st.video("supream.mp4")

    