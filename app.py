import gradio as gr
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
import openai
import os
import datetime

# Load environment variables
app_id = os.environ["WOLFRAM_ALPHA_APPID"]
news_api_key = os.environ["NEWS_API_KEY"]
tmdb_bearer_token = os.environ["TMDB_BEARER_TOKEN"]
openai_api_key = os.environ["OPENAI_API_KEY"]

# Load necessary tools and agents
tool_names = ['serpapi', 'wolfram-alpha', 'pal-math', 'news-api']
llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)
tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Define chat function
def chat(user_input, chat_history):
    # Print the user input and the date and time of the query
    print(f"\n=-=-=-= date and time of query: {datetime.datetime.now()} =-=-=-=")
    print(f"user input: {user_input}")
    # Append the user input and the date and time of the query to the chat history
    chat_history = chat_history or []
    agent_response = agent.run(user_input)
    chat_history.append((user_input, agent_response))
    # Return the chat history to the chatbot and the gradio state
    return chat_history, chat_history

# Define Gradio interface components
message = gr.inputs.Textbox(label="What's your question?", placeholder="What's the answer to life, the universe, and everything?", lines=1)
chatbot = gr.outputs.Chat(label="Chatbot")

examples = [
    "How many people live in Canada?",
    "A triangle has the following side lengths: 4 cm, 4 cm and 4 cm. What kind of triangle is it?",
    "There are 235 books in a library. On Monday, 123 books are taken out. On Tuesday, 56 books are brought back. How many books are there now?",
]

gr.Interface(
    fn=chat,
    inputs=[message, gr.inputs.Chat("Chat History")],
    outputs=[chatbot, gr.outputs.Chat("Chat History")],
    title="LangChain AI",
    description="Ask me anything!",
    examples=examples,
    allow_flagging=False,
    allow_screenshot=False,
    allow_download=False,
).launch()
