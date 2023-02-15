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

# wolfram alpha app_id
app_id = os.environ["WOLFRAM_ALPHA_APPID"]

# news api key
news_api_key = os.environ["NEWS_API_KEY"]

# tmdb bearer token
tmdb_bearer_token = os.environ["TMDB_BEARER_TOKEN"]

# init agent
tool_names = ['python_repl', 'serpapi', 'wolfram-alpha', 'requests', 'terminal', 'pal-math', 'pal-colored-objects', 'llm-math', 'open-meteo-api', 'news-api', 'tmdb-api']
llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key=os.environ["open_ai_api_key"])
tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Define chat function
def chat(inp, history, agent):
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    history = history or []
    output = agent.run(inp)
    history.append((inp, output))
    return history, history

# this is the code for the gradio interface
block = gr.Blocks(css=".gradio-container {background-color: red}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain AI</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(label="What's your question?",
                             placeholder="What's the answer to life, the universe, and everything?",
                             lines=1)
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=["How many people live in Canada?",
                  "What is 13**.3?",
                  "How much did it rain in SF today?",
                  "Get me information about the movie 'Avatar'",
                  "What are the top tech headlines in the US?",
                  "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses - "
                  "if I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"],
        inputs=message
    )

    gr.HTML("""<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>""")

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])

block.launch(debug = True)