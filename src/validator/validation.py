from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from databricks.sdk import WorkspaceClient

# 1. Read entities
# entities_df = spark.table("clinical_scribe.silver_entities")
entities_df = spark.table("healthcare_catalog.silver_zone.extracted_entities")

w = WorkspaceClient()

# 2. Validate with Llama
@udf(returnType=StringType())
def validate_entities_udf(original_text, entities_json):
    prompt = f"""Validate these extracted entities against source text.

SOURCE: {original_text}
ENTITIES: {entities_json}

Return JSON with: accuracy_score (0-1), issues, pass (true/false)
"""

    response = w.serving_endpoints.query(
        name="databricks-meta-llama-3-1-70b-instruct",
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=800,
        temperature=0.0
    )

    return response.choices[0].message.content

# 3. Validate
# validated = entities_df.withColumn("validation", validate_entities_udf(col("parsed.text"), col("entities")))

validated = entities_df.withColumn("validation", validate_entities_udf(col("text"), col("entities")))


# 4. Filter high quality only
high_quality = validated.filter("get_json_object(validation, '$.pass') = 'true'")

# 5. Save
# high_quality.write.format("delta").mode("overwrite").saveAsTable("clinical_scribe.gold_validated")

high_quality.write.format("delta").mode("overwrite").saveAsTable("healthcare_catalog.silver_zone.gold_validated")