import requests
import time
from config2 import HF_API_KEY
from colorama import Fore, Style, init

init(autoreset=True)

DEFAULT_MODEL = "google/pegasus-xsum"

def build_api_url(model_name):
    return f"https://router.huggingface.co/hf-inference/models/{model_name}"

def query(payload, model_name=DEFAULT_MODEL):
    api_url = build_api_url(model_name)
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

def summarize_text(text, min_length, max_length, model_name=DEFAULT_MODEL):
    payload = {
        "inputs": text,
        "parameters": {
            "min_length": min_length,
            "max_length": max_length
        }
    }

    print(Fore.BLUE + Style.BRIGHT + f"\n🚀 Performing summarization with model: {model_name}...")

    result = query(payload, model_name=model_name)

    # 🧠 Handle different response types
    if result is None:
        return None

    # Case 1: Model loading
    if isinstance(result, dict) and "error" in result:
        if "loading" in result["error"].lower():
            print(Fore.YELLOW + "⏳ Model is loading... retrying in 5 seconds.")
            time.sleep(5)
            return summarize_text(text, min_length, max_length, model_name)
        else:
            print(Fore.RED + f"❌ API Error: {result['error']}")
            return None

    # Case 2: Valid response
    if isinstance(result, list) and len(result) > 0:
        if "summary_text" in result[0]:
            return result[0]["summary_text"]

    # Case 3: Unexpected format
    print(Fore.RED + "⚠️ Unexpected API response:")
    print(result)
    return None


if __name__ == "__main__":
    print(Fore.YELLOW + Style.BRIGHT + "Welcome to the Text Summarization Tool! What's your name?")
    user_name = input("Enter your name: ").strip()

    if not user_name:
        user_name = "User"

    print(Fore.GREEN + Style.BRIGHT + f"Hello, {user_name}! Let's get started.")

    print(Fore.YELLOW + Style.BRIGHT + "Enter text to summarize (or type 'exit' to quit):")
    user_text = input("Your text: ").strip()

    if not user_text or user_text.lower() == "exit":
        print(Fore.RED + "⚠️ No text entered. Exiting.")
    else:
        print(Fore.YELLOW + "\nEnter model name (or press Enter for default):")
        model_choice = input("Model name: ").strip()

        if not model_choice:
            model_choice = DEFAULT_MODEL

        print(Fore.YELLOW + "\nChoose summarization style:")
        print("1. Concise")
        print("2. Detailed")

        style_choice = input("Enter 1 or 2: ").strip()

        if style_choice == "2":
            min_length = 80
            max_length = 200
            print(Fore.BLUE + "📘 Generating detailed summary...")
        else:
            min_length = 50
            max_length = 150
            print(Fore.BLUE + "📄 Generating concise summary...")

        summary = summarize_text(user_text, min_length, max_length, model_choice)

        if summary:
            print(Fore.GREEN + Style.BRIGHT + "\n✅ Summary generated successfully!")
            print(Fore.CYAN + Style.BRIGHT + f"\n{summary}")
        else:
            print(Fore.RED + Style.BRIGHT + "⚠️ Failed to generate summary.")