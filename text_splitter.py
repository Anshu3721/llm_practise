from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

print(f"Before splitting: {len(documents)} document")
print(f"Total characters: {len(documents[0].page_content)}")

# Step 2 - Split the documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20
)

chunks = splitter.split_documents(documents)

print(f"\nAfter splitting: {len(chunks)} chunks")
print(f"\nChunk 1:\n{chunks[0].page_content}")
print(f"\nChunk 2:\n{chunks[1].page_content}")
print(f"\nMetadata of chunk 1: {chunks[0].metadata}")