from sentence_transformers import SentenceTransformer
import numpy

model=SentenceTransformer('all-MiniLM-L6-v2')

def prepare_chunks(chunks):
    encoded=model.encode(chunks)
    return encoded

def retrieve(question,chunks,encodings):
    question_encoding=model.encode(question)

    similarities=[]
    for emb,chunks in zip(encodings,chunks):
        score=numpy.dot(emb,question)/numpy.linalg.norm(emb)*numpy.linalg.norm(question)
        similarities.append(score,chunk)

    result=[]

    similarities.sort(reverse=True)

    if similarities[0]<0.3:
        return "don't"

    top_k=3
    i=0
    while(i<top_k):
        result.append(similarities[i])

    return result
