import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

st.set_page_config(
    page_title="Netflix RAG App",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix RAG-app med Gemini, LangChain och Chroma")
st.write("Fråga datasetet om Netflix Originals.")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY saknas. Lägg nyckeln i .env-filen.")
    st.stop()


CSV_PATH = "data/NetflixOriginals_clean.csv"
DB_DIR = "chroma_db"
COLLECTION_NAME = "netflix_originals"


@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"Hittar inte filen: {CSV_PATH}")
        st.stop()

    df = pd.read_csv(CSV_PATH)
    return df


df = load_data()

st.subheader("Dataset preview")
st.dataframe(df.head())


@st.cache_resource
def create_vectorstore(df):
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

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR
    )

    return vectorstore


vectorstore = create_vectorstore(df)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


prompt = ChatPromptTemplate.from_template("""
Du är en hjälpsam dataassistent. Svara bara med hjälp av kontexten från Netflix-datasetet.
Om svaret inte finns i kontexten, säg: "Jag hittar inte svaret i datasetet."

Kontext:
{context}

Fråga:
{question}

Svara på svenska. Var tydlig och nämn gärna filmtitlar, genre, språk, runtime eller IMDB Score när det passar.
""")


def ask_rag(question):
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    answer = llm.invoke(messages)

    return answer.content, docs


st.subheader("Ställ en fråga")

question = st.text_input(
    "Skriv din fråga här:",
    placeholder="Till exempel: Which movies have the highest IMDB score?"
)

if st.button("Fråga"):
    if not question.strip():
        st.warning("Skriv en fråga först.")
    else:
        with st.spinner("Söker i datasetet..."):
            answer, docs = ask_rag(question)

        st.subheader("Svar")
        st.write(answer)

        st.subheader("Källor från datasetet")
        for i, doc in enumerate(docs, start=1):
            st.markdown(
                f"""
**{i}. {doc.metadata.get('title')}**  
Genre: {doc.metadata.get('genre')}  
Language: {doc.metadata.get('language')}  
IMDB Score: {doc.metadata.get('imdb_score')}  
Runtime: {doc.metadata.get('runtime')} minutes
"""
            )