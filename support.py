import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, initialize_agent

PATIENT_DATA_FILE = 'output.txt'
SYSTEM_ROLE = "You are an experienced medical professional analysing patient ICU data, so you have the ability to make diagnoses and evaluations. Provide answers to the following questions to the best of your medical knowledge with the data that you are given about the patient. Make discerning conclusions and diagnoses just with the data provided."