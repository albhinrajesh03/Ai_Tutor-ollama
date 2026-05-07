from sentence_transformers import SentenceTransformer
import faiss
import numpy

model=SentenceTransformer('all-MiniLM-L6-v2')

def prepare_chunks(chunks):
  encoded=model.encode(chunks)
  encoded=encoded/numpy.linalg.norm(encoded, axis=1, keepdims=True)
  encoded=encoded.astype("float32")
  return encoded

def retrieve(question, chunks, encoded):
  question_encoded=model.encode([question])
  question_encoded=question_encoded/numpy.linalg.norm(question_encoded, axis=1, keepdims=True)
  question_encoded=question_encoded.astype("float32")

  dimension=question_encoded.shape[1]
  index=faiss.IndexFlatIP(dimension)
  index.add(encoded)

  scores, indices=index.search(question_encoded,2)

  result=[]
  for i in indices[0]:
    result.append(chunks[i])

  return result
