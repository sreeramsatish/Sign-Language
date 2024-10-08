import speech_recognition as sr
from gtts import gTTS
import os
import pyttsx3
import pygame
import datetime
import requests


def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "I'm sorry, I couldn't understand that."
    except sr.RequestError:
        return "API unavailable."

def generate_response(text):
    text = text.lower()
    
    if "hello" in text:
        return "Hello! How can I help you?"
    elif "how are you" in text:
        return "I'm just a program, so I don't have feelings, but I'm functioning properly!"
    elif "emergency" in text or "help" in text:
        return "If it's an emergency, please call 911 or your local emergency number immediately!"
    elif "fire" in text:
        return "If there's a fire, evacuate the area and call the fire department immediately!"
    elif "medical" in text or "hospital" in text:
        return "For medical emergencies, call 911 or your local emergency medical services."
    elif "police" in text:
        return "For police assistance, dial 911 or your local police department's number."
    elif "time" in text:
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        return f"The current time is {current_time}."
    elif "date" in text:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        return f"Today's date is {current_date}."
    elif "weather" in text:
        return "current weather is a little partially cloudy outside"
    elif "your name" in text or "who are you" in text:
        return "I'm a virtual assistant created by OpenAI. How can I assist you?"
    elif "thank you" in text or "thanks" in text:
        return "You're welcome! Let me know if there's anything else I can help with."
    elif "goodbye" in text or "bye" in text:
        return "Goodbye! Stay safe and take care!"
    else:
        return "I'm sorry, I don't understand that. Please provide more context or ask another question."


def speak_text_with_gtts(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def speak_text_with_pyttsx3(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Say something!")
    input_text = recognize_speech_from_mic(recognizer, microphone)
    print(f"You said: {input_text}")

    response_text = generate_response(input_text)
    print(f"Response: {response_text}")

    # Using gTTS for text-to-speech
    speak_text_with_gtts(response_text)

    # Alternatively, using pyttsx3 for offline text-to-speech
    # speak_text_with_pyttsx3(response_text)

if __name__ == "__main__":
    main()
