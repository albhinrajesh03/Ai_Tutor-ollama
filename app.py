from pdf_loader import load_pdf
from rag import split_text
from llm import ask_llm
from st_rag import prepare_chunks, retrieve

text = load_pdf("book.pdf")
chunks = split_text(text)
encodings=prepare_chunks(chunks)

print("AI Tutor (type 'bye' to exit)")

while True:
    question = input("You: ")

    if question.lower() == "bye":
        print("Goodbye")
        break

    best_chunk = retrieve(question, chunks, encodings)
    context = best_chunk

    prompt = f"""
    You are an expert AI tutor.

    Rules:
    - Explain in simple steps
    - Use examples
    - Ask 1 follow-up question at the end

    Context:
    {context}

    Question:
    {question}
    """

    answer = ask_llm(prompt)

    print("AI:")
    print(answer)
    print("\n" + "-"*50 + "\n")
