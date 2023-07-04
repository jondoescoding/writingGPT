# Python 
import glob
import os
import datetime
from PIL import Image
import requests
from io import BytesIO
# Langchain
from langchain.llms import *
from langchain.chains import SequentialChain
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.memory import SimpleMemory
# Custom Utilities
import helper as hp

# ENVIRONMENT VARIABLES
OPENAI_TOKEN = os.environ.get('openAIToken')
REPLICATE_API_TOKEN = os.environ.get('replicate')
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Setting up the LLM
llmOpenAi = OpenAI(openai_api_key=OPENAI_TOKEN, temperature=0.65, max_tokens=500)

llmText2Img = Replicate(model="ai-forever/kandinsky-2:601eea49d49003e6ea75a11527209c4f510a93e2112c969d548fbb45b9c4f19f")

writer = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: You are a professional freelance copy writer.
    Goal: Write a short 100 word eli5 article for: {topic}. Be succint with your word choice and use examples after the explanation.
    Article: Lets think about this step by step...
    """,
    inputVariables=["topic"],
    output_key="article"
)

editor = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: You are a professional editor.
    Goal: For the given article provide a bulleted list with pieces of feedback correcting sentence structure, grammar and replacing words which a 5 year old may not understand.
    Article: {article}
    Edits:
    """,
    inputVariables=["article"],
    output_key="edits"
)


production = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: You are an expert writer with a decade of experience.
    Goal: Given an article and edits you are to construct a final written piece.
    Article: {article}
    Edits: {edits}
    Final Written Piece:
    """,
    inputVariables=["article", "edits"],
    output_key="production"
)

promptGen = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: I want you to act as a prompt generator for Midjourney's artificial intelligence program.
    Goal: Your job is to provide detailed and creative descriptions that will inspire unique and interesting images from the AI based on the article's theme. For example, you could describe a scene from a futuristic city, or a surreal landscape filled with strange creatures. The more detailed and imaginative your description, the more interesting the resulting image will be.
    Article: {article}
    Prompt: 

    """,
    inputVariables=["article"],
    output_key="prompt"
)

text2img = hp.chain(
    llm=llmText2Img,
    template="{prompt}",
    inputVariables=["prompt"],
    output_key="img"
)

chain = SequentialChain(
    chains=[
        writer,
        editor,
        production,
        promptGen,
        text2img
    ],
    input_variables=["topic"],
    output_variables=["production", "img"],
    verbose=True
)

# The output from the chain
repsonse = list(
    chain({"topic":"How to become a stoic"}).items()
)

# Get the current date
now = datetime.datetime.now().strftime('%Y-%m-%d')

with open(f"..\\writingGPT\\content\\{now}_output.txt", "w", encoding="utf-8") as f:
    # Write the last three key-value pairs to the file, one per line
    for key, value in repsonse:
        # Write the key with '# Key:' prefix and newline character
        f.write(f"\n# Key: {key}\n")
        # Write the value with '# Value' prefix and newline character
        f.write(f"# Value: \n{value} \n")

# Getting the image
getImg = requests.get(repsonse[2][1])

# Open the image using Pillow
img = Image.open(BytesIO(getImg.content))

# Save the image to disk
img.save("..\\writingGPT\\images\\Image.png")
