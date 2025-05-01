import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from pydantic import BaseModel
from FINAL_CODING.get_embedding_function import get_embedding_function
from langchain.text_splitter import RecursiveCharacterTextSplitter

from fastapi import FastAPI, APIRouter
from pymongo import MongoClient
import json

app = FastAPI()
chatRouter = APIRouter()

MAX_TOKENS = 4000  # Define the maximum token limit for the model

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


class QueryRequest(BaseModel):
    query_text: str

@chatRouter.post("/updateISMKeywordsAndCheckList")
async def extract_key_word_for_each_ism_control_update_in_ism_collection_back():
    client = MongoClient("mongodb://localhost:27017/")
    mongo_db = client["algo_compliance_db"]
    collection = mongo_db["e8_ism_controls"]
    documents = collection.find()
    for document in documents:
        # Extract the relevant fields from the document
        actionneeded = document.get("actionneeded", "No Title")
        controlid = document.get("control_id", "No Title")
        intent = document.get("intent", "No intent")
        purpose = document.get("purpose", "No Title")
        keywords = document.get("keywords", "No Title")
        if keywords != "" and len(keywords) > 0:
            print(f"Keywords already exist for control_id {controlid}. Skipping...")
            continue

        prompt = f"""
                I want you to act as a compliance expert. I will provide you with the purpose, intent, and compliance action required for a specific Information Security Manual (ISM) control.

                Your task is to generate a list of **relevant keywords and phrases** that can be used for a **vector similarity search** on the client's security policy document.

                The keywords should:
                - Reflect the **intent** and **purpose** of the control.
                - Help retrieve content that may demonstrate **compliance** with the control.
                - Be concise, meaningful, and semantically rich.
                - Include synonyms or domain-relevant phrases when appropriate.

                Return the output as a **comma-separated list of strings**. Do **not** return any extra text, explanation, or formatting.
 
                Purpose: {purpose}
                Intent: {intent}
                Compliance Action Needed: {actionneeded}
            """
        
        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)
        print(f"Response for control_id {controlid}: {response_text}")
        # Update the document in the MongoDB collection with the new keywords
        collection.update_one({"control_id": controlid}, {"$set": {"keywords": response_text}})
    
 

@chatRouter.post("/query")
async def query_documents(request: QueryRequest):
    query_text = request.query_text
    # hierarchical_query("I want to make my company complient on Authentication process, what are the ISM controls I needs to be aware for multi factor authentication . Give only the ISM controls ?")  # Use asyncio.run to call the async function
    # question_text = "what are the ism controls comes under Providing cybersecurity leadership and guidance ?"
    # hierarchical_query(query_text)  # Use asyncio.run to call the async function
    response = query_rag(query_text)  # Use asyncio.run to call the async function
    return {"response": response}

@chatRouter.post("/query2")
def handle_query(query_text: str):
    # Retrive all the documents in E8 Collection from Mongo DB
    client = MongoClient("mongodb://localhost:27017/")
    mongo_db = client["algo_compliance_db"]
    collection = mongo_db["e8_ism_controls"]
    documents = collection.find()

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


    for document in documents:
        # Extract the relevant fields from the document
        actionneeded = document.get("actionneeded", "No Title")
        keywords = document.get("keywords", "No Title")

        # Search the DB.
        results = db.similarity_search_with_score(keywords, k=4)
        client_Policy_Document = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        #now do compliance check on 'context_text' and 'actionneeded'
        # Combine the results into a single context
        if len(client_Policy_Document) > 5000:
            client_Policy_Document = client_Policy_Document[:5000]
 
        PROMPT_TEMPLATE = f"""
            You are a highly experienced **Information Security compliance expert** with deep knowledge of the Australian ISM framework.

            Your task is to perform a **compliance check** by comparing the client's security policy document against a specific **action requirement** from an ISM control.

            ---

            **Client Policy Document Content**:
            {client_Policy_Document}

            **Action Required by ISM Control**:
            {actionneeded}

            ---

            Please analyze whether the client's policy satisfies the **action required** as described in the ISM control.

            ### Return one of the following:
            - "✅ Compliant" — if the policy explicitly meets or clearly satisfies the action needed.
            - "❌ Not Compliant" — if the policy does not meet the requirement, is vague, or lacks key elements.

            If **Not Compliant**, briefly explain the recommendations for improvement.:
            - What specific requirement is missing or insufficient?
            - What should be added or improved to achieve compliance?

            ---

            ✅ Return the response **as raw JSON only** with the exact structure below. Include only the required fields and values, without any additional text or formatting.
            ❌ Do NOT include any explanations or markdown formatting like triple backticks (```).

            Expected JSON format:
            {{
            "compliance_status": "✅ Compliant" or "❌ Not Compliant",
            "recommendations": "Your recommendations here"
            }}
        """

        model = Ollama(model="llama3.1:8b")
        response_text = model.invoke(PROMPT_TEMPLATE)
        data = json.loads(response_text)
        compleience_Status = data.get("compliance_status", "No compliance status provided")
        recommendations = data.get("recommendations", "No recommendations provided")
        print(f"Response for control_id {document['control_id']}: {response_text}")
        insert_update_e8_complience_details(
            document["control_id"],
            compleience_Status,
            recommendations
        )



def insert_update_e8_complience_details(ismControlName, complianceStatus, recommendations):
    client = MongoClient("mongodb://localhost:27017/")
    mongo_db = client["algo_compliance_db"]
    collection = mongo_db["e8_ism_controls_compliance_details"]

    # Create a new document with the compliance details
    new_document = {
        "ismControlName": ismControlName,
        "complianceStatus": complianceStatus,
        "recommendations": recommendations,
    }

    # Insert the new document into the collection
    collection.insert_one(new_document)

 
def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    print(results)

    # Combine the results into a single context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Split the context if it exceeds the token limit
    if len(context_text) > 8000:
        print("⚠️ Context exceeds token limit. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500,  # Leave some buffer for the question and prompt
            chunk_overlap=500,
        )
        context_chunks = text_splitter.split_text(context_text)
    else:
        context_chunks = [context_text]

    # Process each chunk and combine the results
    combined_response = ""
    for i, chunk in enumerate(context_chunks):
        print(
            f"------>>>>>>-------->>>>Processing chunk {i + 1}/{len(context_chunks)}..."
        )
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=chunk, question=query_text)

        model = Ollama(model="llama3.1:8b")
        response_text = model.invoke(prompt)
        combined_response += f"Chunk {i + 1} Response:\n{response_text}\n\n"

    print(
        f"?????????????????????????????????????????????Final Combined Response:\n{combined_response}"
    )
    return combined_response


def hierarchical_query(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=15)

    # Combine the results into a single context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Split the context if it exceeds the token limit
    if len(context_text) > 8000:
        print("⚠️ Context exceeds token limit. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500,  # Leave some buffer for the question and prompt
            chunk_overlap=500,
        )
        context_chunks = text_splitter.split_text(context_text)

        # Query each chunk and combine the results
        intermediate_responses = []
        for i, chunk in enumerate(context_chunks):
            print(f"Processing chunk {i + 1}/{len(context_chunks)}...")
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=chunk, question=query_text)

            model = Ollama(model="llama3.1:8b")
            response_text = model.invoke(prompt)
            intermediate_responses.append(response_text)

        # Combine intermediate responses into a final query
        combined_context = "\n\n".join(intermediate_responses)
        final_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        final_prompt = final_prompt.format(
            context=combined_context, question=query_text
        )

        print("Querying final combined context...")
        final_response = model.invoke(final_prompt)
        print(
            f" ???????????????????????????????????????????????? Hirarical Final Response:\n{final_response}"
        )
        return final_response
    else:
        # If context fits within the token limit, query directly
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(model="llama3.1:8b")
        response_text = model.invoke(prompt)
        print(
            f" ???????????????????????????????????????????????? Hirarical Final Response:\n{response_text}"
        )
        return response_text

