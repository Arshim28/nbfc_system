import os
import glob
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in .env file")

client = genai.Client(api_key=API_KEY)

def list_files(directory):
    files = glob.glob(os.path.join(directory, '*'))
    return [f for f in files if os.path.isfile(f)]

def select_file(files):
    print("Available files:")
    for idx, f in enumerate(files):
        print(f"{idx + 1}: {os.path.basename(f)}")
    while True:
        try:
            choice = int(input("Select a file by number: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
        except Exception:
            pass
        print("Invalid selection. Try again.")

def get_or_create_cache(file_path, model="gemini-2.5-flash-preview-05-20"):
    cache_name = f"cache_{os.path.basename(file_path)}"
    for cache in client.caches.list():
        if getattr(cache, "display_name", None) == cache_name:
            print(f"Using existing cache: {cache.name}")
            return cache
    print(f"Uploading and caching file: {file_path}")
    uploaded_file = client.files.upload(file=file_path)
    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            contents=[uploaded_file],
            display_name=cache_name,
            ttl="1000s",  
        ),
    )
    print(f"Created cache: {cache.name}")
    return cache

def ask_question(cache, model="gemini-2.5-flash-preview-05-20"):
    while True:
        question = input("Ask a question about the document (or 'exit' to quit): ")
        if question.strip().lower() == 'exit':
            break
        response = client.models.generate_content(
            model=model,
            contents=question,
            config=types.GenerateContentConfig(
                cached_content=cache.name,
            ),
        )
        print("\nResponse:")
        print(response.text)
        usage = getattr(response, 'usage_metadata', None)
        if usage:
            print(f"Token usage: prompt={getattr(usage, 'prompt_token_count', 'N/A')}, response={getattr(usage, 'response_token_count', 'N/A')}, total={getattr(usage, 'total_token_count', 'N/A')}")
        else:
            print("Token usage: Not available")
        print()

def main():
    data_dir = "data"
    files = list_files(data_dir)
    if not files:
        print(f"No files found in {data_dir}/")
        return
    file_path = select_file(files)
    cache = get_or_create_cache(file_path)
    ask_question(cache)

if __name__ == "__main__":
    main() 