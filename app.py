# gradio
import gradio as gr

# langchain
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent

#openai
import openai

# other imports
import os
import datetime
import io
import requests

# # # # # # wolfram alpha app_id
app_id = os.environ["WOLFRAM_ALPHA_APPID"]

# news api key
news_api_key = os.environ["NEWS_API_KEY"]

# tmdb bearer token
tmdb_bearer_token = os.environ["TMDB_BEARER_TOKEN"]

# initialize agent
tool_names = ['serpapi', 'wolfram-alpha', 'pal-math', 'news-api'] # 'open-meteo-api', 'tmdb-api', 'pal-colored-objects'
llm = OpenAI(model_name="text-davinci-003", openai_api_key=os.environ["open_ai_api_key"])
tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Define chat function
def chat(user_input, chat_history):
    # Print the user input and the date and time of the query
    print("\n=-=-=-= date and time of query: " + str(datetime.datetime.now()) + " =-=-=-=")
    print("user input: " + user_input)
    # Append the user input and the date and time of the query to the chat history
    chat_history = chat_history or []
    agent_response = agent.run(user_input)
    chat_history.append((user_input, agent_response))
    # Return the chat history to the chatbot and the gradio state
    return chat_history, chat_history

# create the chatbot
chatbot = gr.outputs.Textbox(label="Conversation", lines=10)

# create the question input
question = gr.inputs.Textbox(label="What's your question?", placeholder="What's the answer to life, the universe, and everything?")

# create the send button
send_button = gr.outputs.Button(label="Send")

# create the initial state
state = gr.outputs.Label(value="Type in a question and click 'Send' to start the conversation.")

# create the gradio interface
interface = gr.Interface(fn=chat, inputs=[question, state, send_button], outputs=[chatbot, state], capture_session=True)

# launch the interface
interface.launch(share=False)