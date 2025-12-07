from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()
history = []

while True:
    user_input = input("Ja: ")

    history.append({"role": "user", "parts": [{"text": user_input}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=history
    )
    assistant_text = response.text
    history.append({"role": "model", "parts": [{"text": assistant_text}]})

    print(f"Gemini: {assistant_text}")