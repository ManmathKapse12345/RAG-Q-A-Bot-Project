from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

if key:
    print(f"✅ API key found (starts with: {key[:12]}...)")
else:
    print("❌ No API key found — check your .env file")