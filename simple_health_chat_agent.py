#!pip install google-generativeai --quiet

import os
from datetime import datetime
import google.generativeai as genai
from google.colab import userdata


# Import API Key
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))
#client = genai.Client(api_key="xxxxxxxxxxxxxxxxxx")

# Setup Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")
chat = model.start_chat()

# How to take user input
user_input = input ("Please describe your health symptoms: ")

# Use Gemini to analyse and respond
response = chat.send_message(f"""The patient says: {user_input}.
#1. Is this a serious medical condition serious?
#2. Severity (High/Medium/Low)
#3. Possible causes?
#4. What should patient do next?
Keep answer simple.
""")

print ("AI advise , not for medical/legal purpose:")
print(response.text)