from openai import OpenAI
from MyKey import MY_OPENAI_KEY

client = OpenAI(api_key=MY_OPENAI_KEY)

response = client.chat.completions.create(model="gpt-4o",
    messages=(
        {"role": "user", "content": "What is today's date?"},
    )
)

print(response.choices[0].message.content)
