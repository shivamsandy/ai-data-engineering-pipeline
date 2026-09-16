import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import snowflake.connector

from dotenv import load_dotenv

load_dotenv(dotenv_path="airflow/.env")


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"

NEW_REVIEWS = 500
TOP_K = 5

CACHE_FILE = "review_embeddings.parquet"

OLLAMA_URL = "http://localhost:11434"


# ============================================================
# READ REVIEWS FROM SNOWFLAKE
# ============================================================

def read_reviews_from_snowflake():

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """

    cursor = conn.cursor()

    try:
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
    finally:
        cursor.close()
        conn.close()

    df.columns = [col.lower() for col in df.columns]

    return df


# ============================================================
# CREATE EMBEDDINGS USING OLLAMA
# ============================================================

def embed(texts):

    embeddings = []

    for text in texts:

        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "prompt": str(text)
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        embeddings.append(data["embedding"])

    return embeddings


# ============================================================
# LOAD REVIEWS + CACHE EMBEDDINGS
# ============================================================

@st.cache_data
def load_reviews():

    if os.path.exists(CACHE_FILE):

        df = pd.read_parquet(CACHE_FILE)

        # Convert stored embeddings back to numpy arrays if needed
        df["embedding"] = df["embedding"].apply(
            lambda x: np.array(x, dtype=float)
        )

        return df

    df = read_reviews_from_snowflake()

    df = df.dropna(subset=["comment"])

    df["embedding"] = embed(
        df["comment"].astype(str).tolist()
    )

    df.to_parquet(
        CACHE_FILE,
        index=False
    )

    return df


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vec_a, vec_b):

    vec_a = np.array(vec_a, dtype=float)
    vec_b = np.array(vec_b, dtype=float)

    denominator = (
        np.linalg.norm(vec_a) *
        np.linalg.norm(vec_b)
    )

    if denominator == 0:
        return 0.0

    return np.dot(vec_a, vec_b) / denominator


# ============================================================
# FIND MOST RELEVANT REVIEWS
# ============================================================

def find_similar_reviews(question, df):

    question_vector = embed([question])[0]

    scores = []

    for review_vector in df["embedding"]:

        score = cosine_similarity(
            question_vector,
            review_vector
        )

        scores.append(score)

    result = df.copy()

    result["score"] = scores

    return result.nlargest(
        TOP_K,
        "score"
    )


# ============================================================
# ASK LLAMA THROUGH OLLAMA
# ============================================================

def ask_llm(question, top_reviews):

    context = ""

    for _, row in top_reviews.iterrows():

        context += (
            f"City: {row['city']}\n"
            f"Rating: {row['rating']} stars\n"
            f"Review: {row['comment']}\n\n"
        )

    system_prompt = """
You are an assistant analyzing customer reviews
for a food delivery application.

Answer ONLY using the customer reviews provided.

Do not invent information.

If the reviews do not contain enough information
to answer the question, say so.

Keep the answer concise and clear.
"""

    user_prompt = f"""
Question:
{question}

Customer Reviews:
{context}
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Zomato Review RAG",
    page_icon="🍽️"
)

st.title("Chat with your Zomato Reviews")

st.caption(
    f"Searching {NEW_REVIEWS} reviews | "
    f"Embedding: {EMBEDDING_MODEL} | "
    f"LLM: {CHAT_MODEL}"
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    review_df = load_reviews()

except Exception as e:

    st.error(
        f"Unable to load reviews: {e}"
    )

    st.stop()


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder=(
        "e.g. What are the most common "
        "complaints about delivery?"
    )
)


# ============================================================
# RAG PIPELINE
# ============================================================

if question:

    try:

        with st.spinner(
            "Searching relevant reviews..."
        ):

            top_reviews = find_similar_reviews(
                question,
                review_df
            )

        with st.spinner(
            "Generating answer with Ollama..."
        ):

            answer = ask_llm(
                question,
                top_reviews
            )

        st.markdown("### Answer")

        st.write(answer)

        with st.expander(
            "Reviews used to build this answer"
        ):

            st.dataframe(
                top_reviews[
                    [
                        "city",
                        "rating",
                        "comment",
                        "score"
                    ]
                ],
                hide_index=True
            )

    except Exception as e:

        st.error(
            f"RAG pipeline failed: {e}"
        )