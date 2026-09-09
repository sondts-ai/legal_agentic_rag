import os

from embeddings import VLLMEmbedding


embedding = VLLMEmbedding(
    api_key=os.getenv("HF_TOKEN"),
    base_url="http://localhost:8080/v1",
    model="intfloat/multilingual-e5-small",
)

texts = [
    f"Đây là nội dung của chunk số {i}"
    for i in range(100)
]

vectors = embedding.embed_documents(texts)

print("Texts:", len(texts))
print("Vectors:", len(vectors))
print("Dimension:", len(vectors[0]))
print("First vector:", vectors[0][:5])

query = "Điều kiện để được cấp giấy phép là gì?"

query_vector = embedding.embed_query(query)

print("Query dimension:", len(query_vector))
print("Query vector:", query_vector[:5])