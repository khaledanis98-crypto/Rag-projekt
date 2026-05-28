import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CSV_PATH = "data/NetflixOriginals_clean.csv"
DB_DIR = "chroma_db"
COLLECTION_NAME = "netflix_originals"

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY saknas. Skapa en .env-fil och lägg in din nyckel.")

def load_csv_as_documents(csv_path: str):
    df = pd.read_csv(csv_path)

    required_columns = ["Title", "Genre", "Premiere", "Runtime", "IMDB Score", "Language"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Saknade kolumner i CSV: {missing}")

    documents = []
    for index, row in df.iterrows():
        text = (
            f"Title: {row['Title']}\n"
            f"Genre: {row['Genre']}\n"
            f"Premiere: {row['Premiere']}\n"
            f"Runtime: {row['Runtime']} minutes\n"
            f"IMDB Score: {row['IMDB Score']}\n"
            f"Language: {row['Language']}"
        )

        metadata = {
            "row": int(index),
            "title": str(row["Title"]),
            "genre": str(row["Genre"]),
            "language": str(row["Language"]),
            "imdb_score": float(row["IMDB Score"]),
            "runtime": int(row["Runtime"]),
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents

def create_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def build_or_load_vectorstore():
    embeddings = create_embeddings()

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )

    documents = load_csv_as_documents(CSV_PATH)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
    )

    return vectorstore

def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template(
        """Du är en hjälpsam dataassistent. Svara bara med hjälp av kontexten från Netflix-datasetet.
Om svaret inte finns i kontexten, säg: "Jag hittar inte svaret i datasetet."

Kontext:
{context}

Fråga:
{question}

Svara på svenska. Var tydlig och nämn gärna filmtitlar, genre, språk, runtime eller IMDB Score när det passar."""
    )

    def rag_answer(question: str):
        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        messages = prompt.format_messages(context=context, question=question)
        answer = llm.invoke(messages)

        return answer.content, docs

    return rag_answer

def main():
    print("Bygger/laddar RAG-systemet...")
    vectorstore = build_or_load_vectorstore()
    rag_answer = create_rag_chain(vectorstore)

    print("\nRAG är redo!")
    print("Skriv en fråga om Netflix-datasetet. Skriv 'exit' för att avsluta.\n")

    while True:
        question = input("Fråga: ").strip()

        if question.lower() in ["exit", "quit", "sluta"]:
            print("Avslutar.")
            break

        if not question:
            continue

        answer, docs = rag_answer(question)

        print("\nSvar:")
        print(answer)

        print("\nKällor från datasetet:")
        for i, doc in enumerate(docs, start=1):
            print(f"{i}. {doc.metadata.get('title')} | Genre: {doc.metadata.get('genre')} | IMDB: {doc.metadata.get('imdb_score')}")

        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()