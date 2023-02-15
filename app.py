import gradio as gr

from langchain.llms import OpenAI

import os

# read in the API key from secrets
open_ai_api_key = os.environ.get('open_ai_api_key')
wolfram_app_id = os.environ.get('wolfram_app_id')

# instantiate the API wrappers
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)

# Define Gradio interface
def chatbot(input_text):
    # Use LangChain to generate response
    response = llm(input_text)
    return response

iface = gr.Interface(fn=chatbot, inputs="text", outputs="text")
iface.launch()