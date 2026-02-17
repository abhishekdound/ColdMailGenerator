import pandas as pd
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



class Portfolio:

    def __init__(self, file_path="resource/my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index_file = "faiss_index.bin"
        self.meta_file = "faiss_meta.pkl"

        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.build_index()

    def build_index(self):
        texts = self.data["Techstack"].tolist()
        embeddings = self.model.encode(texts)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

        self.metadata = self.data["Links"].tolist()

        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def query(self, skills, k=2):
        query_embedding = self.model.encode([skills])
        distances, indices = self.index.search(np.array(query_embedding), k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results
