from sentence_transformers import SentenceTransformer
import numpy
import faiss

model=SentenceTransformer('all-MiniLM-L6-v2')

def prepare_chunks(chunks):
    encoded=model.encode(chunks).astype("float32")
    return encoded

def retrieve(question,chunks,encodings):
    question_encoding=model.encode(question).astype("float32")

    dimension=encodings.shape[1]
    index=faiss.IndexFlatL2(dimension)

    index.add(encodings)

    distance, indices=index.search(question_encoding,2)

    return chunks[indices[0][0]]
