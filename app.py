# gradio
import gradio as gr

# langchain
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, get_all_tool_names

#openai
import openai

# other imports
import os
import datetime
import io
import requests

def set_openai_api_key(api_key, agent):
    if api_key:
        tool_names = get_all_tool_names()

        # load in the api key and initialize gpt3
        llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])

        # # in prod this should look like this so we are taking the api key from the textbox(user input) and charging them for the api calls ### very important ###
        # os.environ["OPENAI_API_KEY"] = api_key
        # llm = OpenAI(model_name="text-davinci-003", temperature=0)
        # os.environ["OPENAI_API_KEY"] = ""

        tools = load_tools(tool_names, llm=llm)
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
        return agent

# Define chat function
def chat(inp, history, agent):
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    history = history or []
    output = agent.run(inp)
    history.append((inp, output))
    return history, history

# this is the code for the gradio interface
block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain AI</center></h3>")

        openai_api_key_textbox = gr.Textbox(placeholder="Paste your OpenAI API key (sk-...)",
               show_label=False, lines=1, type='password')

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

    # if someone changes the API key text box, we set api key and initialize agent and load tools
    openai_api_key_textbox.change(set_openai_api_key,
                                inputs=[openai_api_key_textbox, agent_state],
                                outputs=[agent_state])

block.launch(debug = True)
