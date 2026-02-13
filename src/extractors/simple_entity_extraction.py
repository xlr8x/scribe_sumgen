# 1. Read and collect (good for Community Edition with small data)
parsed_rows = spark.table("healthcare_catalog.raw_zone.parsed_documents").collect()

# 2. Create client OUTSIDE Spark
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()

# 3. Process each document
results = []

for row in parsed_rows:
    text = row.text
    
    prompt = f"""Extract clinical entities from this note as JSON:

{text}

Return JSON with: chief_complaint, symptoms, diagnoses, medications, procedures, vital_signs
"""
    
    try:
        response = w.serving_endpoints.query(
            name="databricks-dbrx-instruct",
            messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
            max_tokens=1500,
            temperature=0.1
        )
        entities = response.choices[0].message.content
    except Exception as e:
        entities = f'{{"error": "{str(e)}"}}'
    
    results.append((row.path, text, entities))

# 4. Create DataFrame from results
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("path", StringType(), True),
    StructField("text", StringType(), True),
    StructField("entities", StringType(), True)
])

extracted = spark.createDataFrame(results, schema)

# 5. Save
extracted.write.format("delta").mode("overwrite").saveAsTable("healthcare_catalog.silver_zone.extracted_entities")
