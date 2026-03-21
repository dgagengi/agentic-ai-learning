# Step1: Install all required packages

# langchain - core framework
# langchain-google-genai - integrate with gemini models
# langchain community - document loaders, utilities
# python-dotenv - load env variables
# pypdf and docx2txt - read pdf & word
# langchain-text-splitters  - split text into chunks
# langchain-classic - LLm Chain
# langchain-core - prompt template

#!pip install -q langchain langchain-google-genai langchain-core python-dotenv pypdf docx2txt langchain-test-splitter langchain-classic lang-chain-community

# Step2: Import libraries

# import os - interact with env variables
# files - upload files
# userdata - access secrets stored in colab

import os
from google.colab import files, userdata


# LLM Wrapper for Google, it allows to call Gemini like Chatbot
from langchain_google_genai import ChatGoogleGenerativeAI

# Library to allow document of diff formats
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Helps to create structure prompt
from langchain_core.prompts import PromptTemplate

# All stps define above are put in chain
from langchain.chains import LLMChain

# Convert output into usable format
from langchain_core.output_parsers import StrOutputParser

# Step3: API Key Setup
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# Step4: LLM & Output Parser
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.8)
output_parser = StrOutputParser()

# Step5: Create Prompt Template
template = """
You are going to work as AI Resume Screener

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

prompt = PromptTemplate(template=template, input_variables=["job_description", "resume_text"])

# Combining all above predefined steps so that it can work together ( chains )
chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)

# Step6: Upload and extract text out of it ( chunks )

print("Please upload your resume (PDF/DOCX/TXT):")
uploaded = files.upload()
if not uploaded:
    raise ValueError("No file uploaded.")

# Extract format of uploaded document
resume_path = list(uploaded.keys())[0]

if resume_path.lower().endswith(".pdf"):
    loader = PyPDFLoader(resume_path)
elif resume_path.lower().endswith(".docx"):
    loader = Docx2txtLoader(resume_path)
else:
    loader = TextLoader(resume_path, encoding="utf-8")

docs = loader.load()


# Split resume(document) into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# Combining all chunks and removing extra spaces and overlap data
resume_text = " ".join(c.page_content for c in chunks).strip()

# Error handling
if not resume_text:
    raise ValueError("Could not extract text from uploaded file.")

# Step7: run analysis and define job description

job_description = """ We are hiring for a Lead Python Developer with experience in Python, AI, SQL, Cloud (AWS/GCP), and Data Analysis."""

result = chain.run(job_description=job_description, resume_text=resume_text)

print("====Result Analysis=====")
print(result)



