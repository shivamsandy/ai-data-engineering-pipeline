import os
import pandas as pd
import streamlit as st
import snowflake.connector
import requests
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path="airflow/.env")

MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434"


FORBIDDEN_WORDS = [
    "drop", "delete", "truncate", "alter",
    "update", "insert", "create", "replace",
    "grant", "revoke"
]

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method"
]


# ============================================================
# SNOWFLAKE SCHEMA
# ============================================================

SCHEMA = """
Tables available in Snowflake:

FCT_ORDERS(
    order_id,
    order_date,
    customer_id,
    restaurant_id,
    city,
    cuisine,
    payment_method,
    order_status,
    is_delivered,
    sales_amount,
    discount,
    delivery_fee,
    gst,
    customer_rating,
    delivery_time_min
)

DIM_RESTAURANTS(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    rating,
    cost_for_two
)

DIM_CUSTOMER(
    customer_id,
    customer_name,
    age,
    age_segment,
    gender,
    city
)

MART_DAILY_CITY_REVENUE(
    order_date,
    city,
    orders,
    cancel_rate,
    gmv,
    aov
)

MART_RESTAURANT_PERFORMANCE(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    orders,
    revenue,
    avg_customer_rating,
    cancel_rate
)

MART_DELIVERY_SLA(
    city,
    order_hour,
    delivered_orders,
    p50,
    p90
)

Important definitions:
- gmv means delivered revenue.
- p50 is the median delivery time.
- p90 is the 90th percentile delivery time.
"""


# ============================================================
# LLM SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a Snowflake SQL expert.

Convert the user's natural-language question into ONE valid
Snowflake SELECT query.

Rules:

1. SELECT queries only. Never modify data.

2. Use ONLY the tables and columns explicitly listed in the
   schema below.

3. Never invent a column.

4. Before choosing a table, verify that every required column
   exists in that table.

5. Use bare table names.
   Correct:
       FCT_ORDERS
   Incorrect:
       ZOMATO.MARTS.FCT_ORDERS

6. Prefer MART tables only when the MART contains ALL columns
   required to answer the question.

7. If the question requires payment_method, use FCT_ORDERS,
   because payment_method exists there.

8. If cancel rate is requested for a grouping that does not
   have a precomputed cancel_rate column, calculate it from
   FCT_ORDERS using order_status.

   Cancel rate formula:

   100.0 * SUM(
       CASE
           WHEN LOWER(order_status) = 'cancelled' THEN 1
           ELSE 0
       END
   ) / NULLIF(COUNT(*), 0)

9. For delivery-time questions:
   - FCT_ORDERS.delivery_time_min is individual delivery time.
   - MART_DELIVERY_SLA.p50 is median delivery time.
   - MART_DELIVERY_SLA.p90 is 90th percentile delivery time.

10. For "average delivery time", use:
       AVG(delivery_time_min)
    from FCT_ORDERS.

11. Add LIMIT 100 or less unless the question asks for
    a single aggregate value.

12. Return ONLY JSON in exactly this format:

{{"sql": "SELECT ..."}}

Schema:

{SCHEMA}
"""


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

@st.cache_resource
def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse="ZOMATO_WH",
        database="ZOMATO",
        schema="MARTS",
        role="DBT_ROLE"
    )


# ============================================================
# GENERATE SQL USING OLLAMA
# ============================================================

def generate_sql(question):

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0
            }
        },
        timeout=180
    )

    response.raise_for_status()

    answer = response.json()["message"]["content"]

    sql = json.loads(answer)["sql"]

    # Remove prefixes if Llama adds them anyway
    sql = (
        sql.replace("ZOMATO.MARTS.", "")
           .replace("ZOMATO.", "")
    )

    return sql.strip().rstrip(";")


# ============================================================
# SQL SAFETY CHECK
# ============================================================

def is_safe(sql):

    lowered = sql.lower().strip()

    if not (
        lowered.startswith("select")
        or lowered.startswith("with")
    ):
        return False

    for word in FORBIDDEN_WORDS:
        if word in lowered:
            return False

    return True


# ============================================================
# RUN QUERY IN SNOWFLAKE
# ============================================================

def run_query(sql):

    conn = get_connection()

    cursor = conn.cursor()

    try:
        return cursor.execute(sql).fetch_pandas_all()

    finally:
        cursor.close()


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Chat with your Zomato Data")

st.caption(
    f"Ask in English, {MODEL} writes the SQL, Snowflake runs it"
)

with st.sidebar:

    st.header("Example Questions")

    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- {q}")


question = st.text_input(
    "Enter your question here",
    placeholder="e.g. Top 10 restaurants by revenue in Bangalore"
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    try:

        sql = generate_sql(question)

        st.subheader("Generated SQL")

        st.code(
            sql,
            language="sql"
        )

        if not is_safe(sql):

            st.error(
                "The generated SQL is not safe to run."
            )

        else:

            df = run_query(sql)

            st.success(
                f"{len(df)} rows returned"
            )

            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True
            )

            # Automatically create chart
            # when result contains 2 columns
            if (
                len(df.columns) == 2
                and len(df) > 0
                and pd.api.types.is_numeric_dtype(
                    df.iloc[:, 1]
                )
            ):

                st.bar_chart(
                    df,
                    x=df.columns[0],
                    y=df.columns[1]
                )

    except requests.exceptions.RequestException as e:

        st.error(
            f"Ollama connection error: {e}"
        )

    except json.JSONDecodeError:

        st.error(
            "Ollama returned invalid JSON. Please try again."
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )