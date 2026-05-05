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

    result = retrieve(question, chunks, encodings)
    if result=="don't":
        print("Bot: I don't know the answer")
        break

    if not result:
        print("Bot: I couldn't find relevant information in the document.")
        break

    context="\n".join(result)

    prompt = f"""
    You are an AI tutor.

    Rules:
        - Answer ONLY using the given context
        - If the answer is not in the context, say "Not found in the document"
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
