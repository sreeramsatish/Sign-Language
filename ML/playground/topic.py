import speech_recognition as sr
import requests

def get_book_info(query):
    GOOGLE_BOOKS_API_ENDPOINT = "https://www.googleapis.com/books/v1/volumes?q="
    response = requests.get(GOOGLE_BOOKS_API_ENDPOINT + query)
    data = response.json()

    if "items" in data:
        book = data["items"][0]
        title = book["volumeInfo"]["title"]
        authors = ", ".join(book["volumeInfo"].get("authors", ["Unknown Author"]))
        description = book["volumeInfo"].get("description", "No description available.")
        return f"Title: {title}\nAuthors: {authors}\nDescription: {description}\n\n"
    else:
        return "No information found for the given book.\n"

def get_topic_info(query):
    WIKIPEDIA_API_ENDPOINT = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    response = requests.get(WIKIPEDIA_API_ENDPOINT)
    data = response.json()

    if "title" in data :
        title = data["title"]
        extract = data["extract"]
        return f"Topic: {title}\nSummary: {extract}\n\n"
    else:
        return "No information found for the given topic.\n"

def main():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Ask about a book or topic...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You asked about: {text}")

            if "book" in text.lower():
                info = get_book_info(text)
            else:
                info = get_topic_info(text)

            print(info)

            # Save the information to a text file
            with open("info.txt", "a") as file:
                file.write(f"Query: {text}\n")
                file.write(info)
                file.write("="*50 + "\n")
            print("Information saved to 'info.txt'")
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError:
            print("API unavailable or unresponsive.")

if __name__ == "__main__":
    main()
