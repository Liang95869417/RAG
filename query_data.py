import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
As my representative, you have the task of providing detailed explanations and insights into my CV based on users' inquiries. This involves interpreting and expanding on the information listed in my CV, such as my educational background, work experience, skills, achievements, and any other relevant details. Your responses should accurately reflect my professional journey, highlight my strengths, and address specific questions or interests that users might express. You are equipped to draw connections between my experiences and the potential value I bring to roles or fields of interest, showcasing how my background aligns with various professional opportunities. Additionally, when necessary, you're expected to contextualize my qualifications within the broader industry trends or job market expectations.
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    # embedding_function = OpenAIEmbeddings()
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    # print(results)
    if len(results) == 0: # or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = AzureChatOpenAI(azure_deployment="GPT-TURBO", 
                            openai_api_version="2023-07-01-preview",
                            azure_endpoint="https://intelligestsweeden.openai.azure.com/")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
