# gradio
import gradio as gr

# old init imports
# langchain
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent

# other imports
import os
import datetime

# # # # # # wolfram alpha app_id
app_id = os.environ["WOLFRAM_ALPHA_APPID"]

# news api key
news_api_key = os.environ["NEWS_API_KEY"]

# tmdb bearer token
tmdb_bearer_token = os.environ["TMDB_BEARER_TOKEN"]

# old way of initializing agent
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

# this is the code for the gradio interface
with  gr.Blocks() as block:
    # create a chatbox to display the bot's response.
    chatbot = gr.Chatbot()
    # create a state to save the chat history.
    state = gr.State()

    with gr.Row():
        #created a textbox so user can input question
        question = gr.Textbox(placeholder="Enter your question here then click 'Send' button or press 'Enter'",
                             lines=1)
        #created a button to send the question to the chatbot.
        send = gr.Button(value="Send", variant="secondary").style(full_width=False)

    # create example questions to show the user the format of the questions they should ask. and what type of question the system is made to handle in both complexity and scope
    gr.Examples(
        examples=["What is the third root of 5249 to the nearest integer?",
                  "A triangle has the following side lengths: 4 cm, 4 cm and 4 cm. What kind of triangle is it?",
                  "There are 235 books in a library. On Monday, 123 books are taken out. On Tuesday, 56 books are brought back. How many books are there now?",
                   "How many kilometers are in a light-year, and what is the square root of that number?",
                    "if 3x-y=12, what is the value of (8^(x)/2^(y))? A) 2^(12) B) 4^(4) C) 8^(2) D) the value cannot be determined from the information given",
                     "The graph of which of the following equations is a straight line parallel to the graph of y = 2x ? a) 4x – y = 4 b) 2x – 2y = 2 c) 2x – y = 4 d) 2x + y = 2 e) x – 2y = 4",
                      "a real estate agent recived a 6% commission on the selling price of a house. if his commission was 8,880 what was the selling price of the house?",
                       "An airplane flies against the wind from A to B in 8 hours. The same airplane returns from B to A, in the same direction as the wind, in 7 hours. Find the ratio of speed of the airplane (in still air) to the speed of the wind."],
        inputs=question
    )

    #connects the send button to the function that will respond to the user's question.
    send.click(chat, inputs=[question, state], outputs=[chatbot, state])
    #connects the enter key to the function that will respond to the user's question.
    question.submit(chat, inputs=[question, state], outputs=[chatbot, state])

    gr.HTML("""<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>""")

block.queue() # was launch(debug = True)

demo = gr.TabbedInterface(

    [block], ["chat"],
    title='Gestalt',

)

demo.queue()
demo.launch(share=False)