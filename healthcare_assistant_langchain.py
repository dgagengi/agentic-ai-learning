# Description: Implementing Healthcare Assitant using LangChain framework

# -- Step1: Installing dependencies/packages
#!pip install -q langchain langchain-google-genai langchain-community python-dotenv langchain-text-splitters langchain-classic langchain-core

# -- Step2: Importing packages/libraries
import os
from datetime import datetime
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

# -- Step3: Configuring API Key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# -- Step4: Configuring LLM & Output Parser
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.1)
output_parser = StrOutputParser()

# -- Step5: Calling LLM
def health_assistant(input_text):
  print ("\n [Health Assistant]")

  # Creating a template
  template = """
  You are an expert Healthcare specialist.
  Your job is to provide correct, precise and short answers for health related questions.
  Do not respond to any other question outside of health related questions.

  {health_question}
  """

  # Creating a prompt using prompt template. Helps structure, reuse, and control how we talk to an LLM.
  prompt = PromptTemplate(template=template, input_variables=["health_question"])

  # Combining all above predefined steps so that it can work together ( chains )
  chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)

  result = chain.run(health_question=input_text)
  return result


# Asking user about symptoms
name = input("Enter patient name: ")
pid = input("Enter patient ID: ")
user_input = input("Please describe your health symptoms: ")

# Understanding health symptoms severity
print ("\n [Health Assistant] --- Understanding health symptoms severity...")
symptom_severity_prompt = f"Classify the severity of this symptom and list possible conditions: {user_input}"
symptom_severity_summary = health_assistant(symptom_severity_prompt)
print (f"Symptom Severity Summary: {symptom_severity_summary}")

# Health symptom analysis
print ("\n [Health Assistant] --- Health symptom analysis...")
symptom_analysis_prompt = f"Based on the summary, suggest next steps for the patient: {symptom_severity_summary}"
symptom_analysis_summary = health_assistant(symptom_analysis_prompt)
print (f"Symptom Analysis Summary: {symptom_analysis_summary}")

# Follow-up questions
print ("\n [Health Assistant] --- Follow-up questions...")
follow_up_questions_prompt = f"What follow-up questions should doctor ask to understand patient condition better?"
follow_up_questions_summary = health_assistant(follow_up_questions_prompt)
print (f"Follow-up Questions Summary: {follow_up_questions_summary}")

# Logging the interaction
print ("\n [Health Assistant] --- Logging the interaction...")
log_entry = f"\nTime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \nPatient: {name} \nPatient ID: {pid}\nInput: {user_input}\nOutput: {symptom_analysis_summary}\n{'=' *50}\n"
with open("patient_log.txt", "a") as file:
    file.write(log_entry)

