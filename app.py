from flask import Flask, render_template, request, jsonify
import openai
import os
from support import *


app = Flask(__name__)

# Set up OpenAI API credentials
#openai.api_key = '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'

global qa

def setup_qa_retrieval_chain():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    loader = UnstructuredFileLoader(PATIENT_DATA_FILE)
    document = loader.load()
    texts = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        max_tokens=1000,
    )
    conversation_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    tools = [
        Tool(
            name="Knowledge Base",
            func=qa.run,
            description="A knowledge base of ICU patient information including procedures, treatments, medications administered, input events, etc. Use this to answer questions about the patient's medical history and to make discerning conclusions and evaluations.",
        )
    ]
    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        max_iterations=5,
        early_stopping_method="generate",
        memory=conversation_memory
    )
    
    return qa

# Define the default route to return the index.html file
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api", methods=["POST"])
def api():
    # Get the message from the POST request
    message = request.json.get("message")
    answer = qa.run(SYSTEM_ROLE + message)
    return jsonify({'text': answer})

if __name__=='__main__':
    os.environ['OPENAI_API_KEY'] = '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'
    qa = setup_qa_retrieval_chain()
    app.run()
    

