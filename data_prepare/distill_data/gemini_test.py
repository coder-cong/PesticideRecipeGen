from google import genai
from google.genai import types

API_KEY = "AIzaSyAqCPv4_z1NC9qD4Z4ZjSMJRguoymjfDto"

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="请介绍一下大模型多智能体系统的最新进展",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True
        )
    )
)

for part in response.candidates[0].content.parts:
    if not part.text:
        continue
    if part.thought:
        print("Thought summary:")
        print(part.text)
        print()
    else:
        print("Answer:")
        print(part.text)
        print()
