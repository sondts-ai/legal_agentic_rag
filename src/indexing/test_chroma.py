import chromadb

client = chromadb.HttpClient(
    host="localhost",
    port=8000,
)

print("Heartbeat:", client.heartbeat())

print("Collections:")
for collection in client.list_collections():
    print("-", collection.name)