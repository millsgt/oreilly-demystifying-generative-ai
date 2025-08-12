import openai
from MyKey import MY_OPENAI_KEY

class PoliteAgent:
    def __init__(self):
        # Store conversation history
        self.conversation_history = [
            {"role": "system",
             "content": """You are a rewrite assistant to suggest more positive messages. 
             
             You will take the user input, which may sometimes be impolite, 
             that is about to be submitted on a social media site. 
             If it is impolite, make it polite and diplomatic. 
             
             For example, if the input is "This is the stupidest product I have ever seen! 
             It not only is overpriced but it is as useless as a bag of rocks!" instead suggest
            "The price and features of this product surprises me. What is the value proposition? 
            Perhaps I'm missing something." 
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
