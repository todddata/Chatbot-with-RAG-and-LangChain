from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db" 

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the model
# llm = ChatOpenAI(temperature=0.5, model='gpt-4o')

# this is the second model I'm trying; it's lower cost to run
llm = ChatOpenAI(temperature=1, model='o4-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 20
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
# chatbot = gr.ChatInterface(
#     fn=stream_response,
#     textbox=gr.Textbox(placeholder="Send to the LLM..."),
#     chatbot=gr.Chatbot(value=[("Research Assistant", "Type a question to learn more about the Weigh and Win Community Weight Loss research study.")]),
#     fill_height=True,
#     autoscroll=True,
# )

# launch the Gradio app
# chatbot.launch(share=True)

import gradio as gr

with gr.Blocks() as demo:
    with gr.Column():
        gr.Image("logo.png", height=258, width=500, show_label=False)
        gr.ChatInterface(
            fn=stream_response,
            textbox=gr.Textbox(placeholder="Send to the LLM..."),
            chatbot=gr.Chatbot(value=[("Chat with this JGIM Research Assistant!❤️", "Type a question to learn more about community weight loss.")]),
            fill_height=True,
            autoscroll=True,
        )

demo.launch(share=True)