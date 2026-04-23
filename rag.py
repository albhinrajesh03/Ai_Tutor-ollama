def split_text(text, chunk_size=800):
    """
    Break large text into smaller chunks for processing
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def retrieve(question, chunks):
    """
    Simple keyword-based retrieval (mini RAG)
    Finds chunks that match question words
    """
    question_words = question.lower().split()
    results = []

    for chunk in chunks:
        chunk_lower = chunk.lower()

        if any(word in chunk_lower for word in question_words):
            results.append(chunk)

    return results[:3]