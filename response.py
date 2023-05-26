import os


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQA

if __name__ == '__main__':
    os.environ['OPENAI_API_KEY'] = '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'

    DOC = 'output.txt'

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    print("1")

    loader = UnstructuredFileLoader(DOC, strategy="fast")

    print("2")

    document = loader.load()

    print("3")

    texts = text_splitter.split_documents(document)

    print(len(texts))

    embeddings = OpenAIEmbeddings()

    #vectors = Chroma.from_documents(texts, embeddings)

    llm = OpenAI()
    db = FAISS.from_documents(texts, embeddings)
    #docs = db.similarity_search(query)

    #print(docs[0].page_content)
    

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())

    print("done")

    print(qa.run("You are a doctor analysing ICU data. Answer the following questions given the provided data and your medical knowledge. How often does the patient recieve insluine injections and how much in units? Is this treatment fine?"))





