from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vecpre_compine import retriever

model = OllamaLLM(model="gemma2:9b")

template = """
    You are an exeprt in answering questions about a vehicle, however you also some provided context of external knowledge not in the scope of car, 
    so if user ask question relevant to the provided context then look into the context to answer user question.
    
    if they ask for api then prioritize giving the whole api and you should you "." in api not "/", however answer as straight to the point as posible
    If they want to change the state of the vehicle or status of vehicle attachment tool then answer arcoding to the example
    Here are some knowledge you need to know about youself: {name}
    Here are some relevant infor: {information}

    Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def ask_and_answer(question, name) :
    information = retriever.invoke(question)
    result = chain.invoke({"information": information, "question": question, "name": name})

    return result