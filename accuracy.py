from sentence_transformers import SentenceTransformer
from sklearn.metrics import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def accuracy(input, target):
    sen1 = [input]
    sen2 = [target]

    sen1 = model.encode(sen1)
    sen2 = model.encode(sen2)

    return cosine_similarity(sen1, sen2)

