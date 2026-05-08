from sentence_transformers import SentencetTransformer
import chromadb

model=SentenceTransformer("all-MiniLM-L2-v2")

client=chromadb.PersistentClient(path="./chroma_db")

collection=client.get_or_create_collection(name="pdf_rag")

def split_text(text, chunk_size=500, overlap=100):
  chunks=[]
  i=0
  while i<len(text):
    chunk=text.split(i:chunk_size)
    chunks.append(chunk)
    i=i+chunk_size-overlap
  return chunks

def prepare_chunk(chunks):
  encoded=chunks.enode(chunks).tolist()
  ids=[]
  for i in range(len(chunks)):
    ids.append(str(i))
  collection.add(documents=chunks, embeddings=encoded, ids=ids)

def retrieve(question):
  question_encoded=model.encode(question).tolist()
  

