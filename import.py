from main import ask_and_answer

while True:
    print("---------------------------")
    question = input("User: ")
    if question == "exit":
        break
    kkk = ask_and_answer(question)
    print(kkk)