from langchain_community.document_loaders import TextLoader

# load the documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

print(f"Number of documents loaded: {len(documents)}")
print(f"Content preview:\n{documents[0].page_content[:200]}")
print(f"Metadata: {documents[0].metadata}")