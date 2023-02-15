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

# initialize agent
tool_names = ['python_repl', 'serpapi', 'wolfram-alpha', 'requests', 'terminal', 'pal-math', 'pal-colored-objects', 'llm-math', 'open-meteo-api', 'news-api', 'tmdb-api']
llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key=os.environ["open_ai_api_key"])
tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)
agent = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True)

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

# this is the code for the gradio interface
block = gr.Blocks()

with block:
    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(label="What's your question?",
                             placeholder="What's the answer to life, the universe, and everything?",
                             lines=1)
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=["How many people live in Canada?",
                  "What is 13**.3?",
                  "what is the weather like today on the College of charleston Campus?",
                  "Get me information about the movie 'Avatar'",
                  "What are the top tech headlines in the US?",
                  "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses - "
                  "if I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"],
        inputs=message
    )

    gr.HTML("""<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>""")

    state = gr.State()

    submit.click(chat, inputs=[message, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state], outputs=[chatbot, state])

block.queue()   # was launch(debug = True)

demo = gr.TabbedInterface(

    [block], ["chat"],
    title='LangChain AI',

)

demo.queue()
demo.launch(share=False)