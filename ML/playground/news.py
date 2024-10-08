import speech_recognition as sr
import requests
import json

def get_latest_news():
    # Replace with your own NewsAPI key
    API_KEY = 'b868a76ded634fc8ab113612da07f619'
    ENDPOINT = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=' + API_KEY
    response = requests.get(ENDPOINT)
    news_data = response.json()

    if news_data["status"] == "ok":
        articles = news_data["articles"]
        news_list = []
        for article in articles:
            title = article["title"]
            description = article["description"]
            news_list.append(f"Title: {title}\nDescription: {description}\n\n")
        return news_list
    else:
        print("Error fetching news")
        return []

def main():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            if "news" in text.lower():
                news_list = get_latest_news()
                with open("latest_news.txt", "w") as file:
                    for news in news_list:
                        file.write(news)
                print("Latest news titles and descriptions saved to 'latest_news.txt'")
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError:
            print("API unavailable or unresponsive.")

if __name__ == "__main__":
    main()
