import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

#print(os.environ.get("GEMINI_API_KEY"))



# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Cześć!"
)
print(response.text)