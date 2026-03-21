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

#Agent 1: Symptom agent
def symptom_agent(input_text):
  print ("\n [Symptom Agent]")
  response = model.generate_content(f"Classify the severity of this symptom and list possible conditions: {input_text}")
  print (response.text)
  return response.text

#Agent 2: Scheduler agent
def scheduler_agent(symptom_severity_summary):
  print("\n [Scheduler Agent]")
  response = model.generate_content(f"Based on the summary, suggest next steps for the patient: {symptom_severity_summary}")
  print (response.text)
  return response.text

#Agent 3: Follow up agent
def follow_up_agent():
  print("\n [Follow Up Agent]")
  response = model.generate_content(f"What follow-up questions should doctor ask to understand patient condition better?")
  print (response.text)
  return response.text

#Agent 4: Logging agent
def logger_agent(name, pid, input_text, output_text):
  print ("\n [Logger Agent]")
  log_entry = f"\nTime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \nPatient: {name} \nPatient ID: {pid}\nInput: {input_text}\nOutput: {output_text}\n{'=' *50}\n"
  with open("patient_log.txt", "a") as file:
    file.write(log_entry)

#Define user inputs
name = input("Enter patient name: ")
pid = input("Enter patient ID: ")
user_input = input("Please describe your health symptoms: ")

#Agent1: Symptom understanding
summary = symptom_agent(user_input)

#Agent2: Scheduler
next_steps = scheduler_agent(summary)


#Agent3: Follow up
followups = follow_up_agent()

#Agent4: Logging
logger_agent(name, pid, user_input, summary)





