import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
from langchain.text_splitter import RecursiveCharacterTextSplitter

MAX_TOKENS = 4000  # Define the maximum token limit for the model

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    #hierarchical_query("I want to make my company complient on Authentication process, what are the ISM controls I needs to be aware for multi factor authentication . Give only the ISM controls ?")  # Use asyncio.run to call the async function
    question_text = "  what is this ISM-1241 control related ? "
    hierarchical_query(question_text)  # Use asyncio.run to call the async function
    query_rag(question_text) # Use asyncio.run to call the async function


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Combine the results into a single context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Split the context if it exceeds the token limit
    if len(context_text) > 8000:
        print("⚠️ Context exceeds token limit. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500,  # Leave some buffer for the question and prompt
            chunk_overlap=500
        )
        context_chunks = text_splitter.split_text(context_text)
    else:
        context_chunks = [context_text]

    # Process each chunk and combine the results
    combined_response = ""
    for i, chunk in enumerate(context_chunks):
        print(f"------>>>>>>-------->>>>Processing chunk {i + 1}/{len(context_chunks)}...")
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=chunk, question=query_text)

        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)
        combined_response += f"Chunk {i + 1} Response:\n{response_text}\n\n"

    print(f"?????????????????????????????????????????????Final Combined Response:\n{combined_response}")
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
            chunk_overlap=500
        )
        context_chunks = text_splitter.split_text(context_text)

        # Query each chunk and combine the results
        intermediate_responses = []
        for i, chunk in enumerate(context_chunks):
            print(f"Processing chunk {i + 1}/{len(context_chunks)}...")
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=chunk, question=query_text)

            model = Ollama(model="mistral")
            response_text = model.invoke(prompt)
            intermediate_responses.append(response_text)

        # Combine intermediate responses into a final query
        combined_context = "\n\n".join(intermediate_responses)
        final_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        final_prompt = final_prompt.format(context=combined_context, question=query_text)

        print("Querying final combined context...")
        final_response = model.invoke(final_prompt)
        print(f" ???????????????????????????????????????????????? Hirarical Final Response:\n{final_response}")
        return final_response
    else:
        # If context fits within the token limit, query directly
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)
        print(f" ???????????????????????????????????????????????? Hirarical Final Response:\n{response_text}")
        return response_text
    

if __name__ == "__main__":
    main()
