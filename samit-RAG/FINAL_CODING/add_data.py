import argparse
import json
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from FINAL_CODING.get_embedding_function import get_embedding_function
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
import uuid
from bson.binary import Binary
from pymongo import MongoClient
from datetime import datetime
from fastapi import FastAPI, APIRouter, UploadFile
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

client = MongoClient("mongodb://localhost:27017/")
mongo_db = client["IRAP_DB"]
file_collection = mongo_db["UserFiles"]

app = FastAPI()
uploadRouter = APIRouter()

CHROMA_PATH = "chroma"
DATA_PATH = "./DATA"


@uploadRouter.post("/upload")
async def upload_documents(files: list[UploadFile]):
    # file_ids_map = {}

    # for file in files:
    #     file_content = await file.read()
    #     filename = file.filename
    #     file_id = f"{filename}:{uuid.uuid4().hex[:8]}"
    #     file_ids_map[filename] = file_id

    #     # Insert into MongoDB
    #     file_doc = {
    #         "file_id": file_id,
    #         "filename": filename,
    #         "upload_time": datetime.utcnow(),
    #         "file_data": Binary(file_content),
    #     }
    #     file_collection.insert_one(file_doc)

    # os.makedirs(DATA_PATH, exist_ok=True)
    # for file in files:
    #     file_path = os.path.join(DATA_PATH, file.filename)
    #     with open(file_path, "wb") as f:
    #         f.write(await file.read())
    # Check if the database should be cleared (using the --clear flag).
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--reset", action="store_true", help="Reset the database.")
    # args = parser.parse_args()
    # if args.reset:
    #     print("✨ Clearing Database")
    #     clear_database()
    reset_flag = os.getenv("RESET_DATABASE", "false").lower() == "true"
    if reset_flag:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    chunks = add_file_ids_to_chunks(chunks, "file_ids_map")

    for chunk in chunks:
        chunk.metadata = clean_metadata(chunk.metadata)
    add_to_chroma(chunks)


# def load_documents():
#     documents = []
#     for filename in os.listdir(DATA_PATH):
#         if filename.endswith(".pdf"):
#             file_path = os.path.join(DATA_PATH, filename)
#             loader = UnstructuredPDFLoader(file_path, mode="paged")
#             docs = loader.load()
#             file_id = f"{filename}:{uuid.uuid4().hex[:8]}"
#             for i, doc in enumerate(docs):
#                 doc.metadata["source"] = file_path
#                 doc.metadata["page_number"] = i + 1
#                 doc.metadata["file_id"] = file_id
#                 doc.metadata["id"] = filename  # just the file name
#             documents.extend(docs)
#     return documents


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_file_ids_to_chunks(chunks: list[Document], file_ids_map: dict):
    for chunk in chunks:
        source_path = chunk.metadata.get("source", "")
        # filename = os.path.basename(source_path)
        # file_id = file_ids_map.get(filename, os.path.splitext(filename)[0])
        chunk.metadata["file_id"] = "test_file_id" + str(uuid.uuid4())
        chunk.metadata["id"] = "id_"+str(uuid.uuid4())  # Generate a unique ID for each chunk
    return calculate_chunk_ids(chunks)


def clean_metadata(metadata):
    cleaned = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            try:
                cleaned[k] = json.dumps(v)
            except Exception:
                cleaned[k] = str(v)
    return cleaned


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        for i in range(0, len(new_chunks), 50):  # Process in batches of 50
            batch = new_chunks[i : i + 50]
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            db.add_documents(batch, ids=batch_ids)
            print(f"✅ Added batch {i // 50 + 1} with {len(batch)} documents")
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
