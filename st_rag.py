from sentence_transformers import SentenceTransformer
import numpy

model=SentenceTransformer('all-MiniLM-L6-v2')

def prepare_chunks(chunks):
    encoded=model.encode(chunks)
    return encoded

def retrieve(question,chunks,encodings):
    question_encoding=model.encode(question)

    similarities=[]
    for emb in encodings:
        score=numpy.dot(emb,question)/numpy.linalg.norm(emb)*numpy.linalg.norm(question)
        similarities.append(score)

    best_index=int(numpy.argmax(similarities))

    return chunks[best_index]
