import openai
from MyKey import MY_OPENAI_KEY


class PoliteAgent:
    def __init__(self):
        # Store conversation history
        self.conversation_history = [
            {"role": "system",
             "content": """
             PUT YOUR PROMPT HEADER INSTRUCTIONS HERE 
             """}
        ]

    def get_response(self, user_input):
        # Add user's message to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            # Make API call to OpenAI
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history,
                max_tokens=250,  # Limit response length
                temperature=0.7  # Controls creativity (0-1)
            )

            # Get AI's response
            ai_response = response.choices[0].message.content
            return ai_response

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"


# Create instance of our AI agent
agent = PoliteAgent()

print("Welcome to Polite Agent!")
print("Provide a hostile social media post and I will make it more diplomatic.")
print("Type 'quit' to exit")

while True:
    # Get user input
    user_input = input("\nYou: ")

    # Check if user wants to quit
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break

    # Get and display AI response
    response = agent.get_response(user_input)
    print(f"AI: {response}")
