import os
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
model_id = 'gemini-2.5-flash-preview-05-20'
search_tool = Tool(
    google_search = GoogleSearch()
)

response = client.models.generate_content(
    model=model_id,
    contents="When is the next eclipse, in India?",
    config=GenerateContentConfig(
        tools=[search_tool],
        response_modalities=["TEXT"]
    )
)

print(response.text)
