from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from prevector import retriever

model = OllamaLLM(model="gemma:7b")

template = """
    You are an Q/A machine, you are Anna, your mission is to answer questions from user.
    If user ask question that you don't know then go to relevant document and find necessary informations there. When you use provided context to 
    answer user question, please don't say 'based on the document provided' or something similar to that, just answer the question
    If user ask question that are not included inside of the document then consider answer it yourself using your own knowledge, and even when you
    can't answer user's questions even when used provided context and your own then just say 'Sorry but I don't know'. 
    Here are some relevant infor: {information}

    Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("User: ")
    if question == "exit":
        break
    information = retriever.invoke(question)
    result = chain.invoke({"information": information, "question": question})
    print("AI: ", result)