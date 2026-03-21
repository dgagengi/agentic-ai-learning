!pip install pypdf docx2txt --quiet
import os
import google.generativeai as genai
from datetime import datetime
from google.colab import files, userdata

# Libraries for direct document parsing
import pypdf
import docx2txt

# Import API Key
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))

# Setup Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Function to upload resume Upload
def upload_resume():
  print("Please upload your resume (PDF/DOCX/TXT):")
  uploaded = files.upload()
  if not uploaded:
    raise ValueError("No file uploaded.")
  # Extract format of uploaded document
  resume_path = list(uploaded.keys())[0]
  return resume_path

# Function to extract text from different file types
def extract_text_from_resume(resume_path):
    text = ""
    if resume_path.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(resume_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif resume_path.lower().endswith(".docx"):
        text = docx2txt.process(resume_path)
    elif resume_path.lower().endswith(".txt"):
        with open(resume_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT.")
    return text

# Agent: Review agent
def review_agent(full_prompt):
  response = model.generate_content(full_prompt)
  return response.text

# Defining Job description.
job_description = """ We are hiring for a Lead Python Developer with experience in Python, AI, SQL, Cloud (AWS/GCP), and Data Analysis."""

# Defining the prompt for the model
prompt = """You are an expert AI Resume Screener

Job Description:
{job_description}

Resume Text:
{resume_text}

Give a simple analysis:
- FIT Score (0- 100)
- Top 5 matching skills
- Missing Important Skills
- One-line Verdict
"""

# Invoking resume upload
resume_path = upload_resume()

# Extracting text from resume
resume_text = extract_text_from_resume(resume_path)

# Constructing prompt for the model
full_prompt = prompt.format(job_description=job_description, resume_text=resume_text)

# Calling the model for screening the resume
response = review_agent(full_prompt)

# Printing the response
print(response)

