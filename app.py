from pdf_loader import load_pdf
from rag import split_text, retrieve
from llm import ask_llm

# Load PDF
text = load_pdf("book.pdf")
chunks = split_text(text)

print("AI Tutor Ready (type 'exit' to quit)")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye")
        break

    relevant_chunks = retrieve(question, chunks)
    context = "\n".join(relevant_chunks[:2])

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