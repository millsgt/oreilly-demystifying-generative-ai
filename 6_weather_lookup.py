import json
from openai import OpenAI
from MyKey import MY_OPENAI_KEY # Replace with your actual OpenAI API key in MyKey.py

# Initialize the OpenAI client
client = OpenAI(api_key=MY_OPENAI_KEY)

# Define the tools (functions) available to the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature (default: fahrenheit)",
                    },
                },
                "required": ["location"],
            },
        },
    }
]


# Mock function to simulate getting weather data (in a real scenario, this would call a weather API like OpenWeatherMap)
def get_current_weather(location, unit="fahrenheit"):
    # Simulated weather data
    if unit == "celsius":
        temperature = "22"
    else:
        temperature = "72"

    weather_info = {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


# Loop to continuously prompt and respond
while True:
    # Get user query from command line input
    user_query = input("Enter your prompt (or 'exit' to quit): ")

    if user_query.lower() == 'exit':
        break

    # Initial messages
    messages = [{"role": "user", "content": user_query}]

    # First API call to check if a tool needs to be called
    response = client.chat.completions.create(
        model="gpt-4o",  # Use a model that supports tool calls, like gpt-4o or gpt-3.5-turbo
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Let the model decide if a tool is needed
    )

    # Extract the response message
    response_message = response.choices[0].message

    # Check if the model wants to call a tool
    tool_calls = response_message.tool_calls

    if tool_calls:
        # Append the assistant's message (which includes the tool call) to the conversation
        messages.append(response_message)

        # Process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Call the corresponding function
            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit")
                )

            # Append the tool's response to the messages
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        # Second API call to get the final response after providing tool results
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        # Print the final assistant response
        print(second_response.choices[0].message.content)
    else:
        # If no tool call, just print the direct response
        print(response_message.content)