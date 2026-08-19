// PROJECT EXPLAINATION 

I built this project to understand how a real-world, cloud-based data engineering pipeline works from end to end.**

The pipeline starts with **raw Zomato CSV data**, which I store in **AWS S3 as the data lake**. From S3, the data is ingested into **Snowflake**, which acts as the cloud data warehouse.

Inside Snowflake, I follow a **layered architecture — RAW, STAGING, MARTS, and AI — so that raw data is kept separate from cleaned and business-ready data.**

For transformations, I use **dbt** to clean the data, apply business logic, create reusable models, and perform data-quality tests.

I use **Apache Airflow for orchestration**, so the complete workflow can run automatically instead of executing every step manually.

The overall architecture is:

**Raw Data → AWS S3 → Snowflake → dbt → Airflow → Analytics / AI-ready Data**

The project helped me understand not just individual tools, but also **cloud integration, data ingestion, warehouse design, transformation, orchestration, security using IAM, and handling real implementation challenges such as cross-region architecture.**

So, this is not simply a data-loading project; it is an **end-to-end cloud data engineering pipeline designed around production-style concepts.”**



Q1. What was the first challenge you faced in your Zomato Data Engineering project?

Answer:
The first challenge I faced was a region mismatch between AWS S3 and Snowflake. My S3 bucket was created in the Mumbai region, but Snowflake Trial did not provide Mumbai as an available region. So I created the Snowflake account in Singapore and planned the integration using an AWS IAM Role, Snowflake Storage Integration, and External Stage. Since my dataset was small, I decided not to recreate the S3 bucket. I also considered the possible cross-region data transfer cost and latency.