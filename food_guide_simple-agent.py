from google import genai
from google.genai.types import GenerateContentConfig

# Configured the API Key in Google AI Studio
client = genai.Client(api_key=userdata.get('GOOGLE_API_KEY'))
#client = genai.Client(api_key="xxxxxxxxxxxxxxxxxx")

SYSTEM_PROMPT = """
You are a helpful assistant that only answers questions about food.
If a user asks a question that is NOT related to food, you MUST respond
with the exact phrase: 'Sorry please ask only questions related to food.'
Do not provide any other information or answer for non-food related queries.
"""
def simple_agent(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig( 
            system_instruction=SYSTEM_PROMPT
        )
    )
    return response.text


print("Welcome to the Food Agent! Ask me anything about food. Type 'exit' to quit.")

while True:
    user_question = input("Your question: ")

    if user_question.lower() == "exit":
        print("Exiting Food Agent. Goodbye!")
        break

    agent_response = simple_agent(user_question)
    print("Agent:", agent_response)