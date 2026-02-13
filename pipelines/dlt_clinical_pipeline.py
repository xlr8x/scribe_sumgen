import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *
from src.extractors.entity_extractor import ClinicalEntityExtractor

# Initialize extractor
extractor = ClinicalEntityExtractor()

# Define extraction UDF
@udf(returnType=StringType())
def extract_entities_udf(clinical_text):
    """UDF wrapper for entity extraction"""
    try:
        entities = extractor.extract_entities(clinical_text)
        return json.dumps(entities)
    except Exception as e:
        return json.dumps({"error": str(e)})

# BRONZE TABLE: Raw parsed documents
@dlt.table(
    name="bronze_parsed_documents",
    comment="Raw parsed clinical documents from AutoLoader",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true"
    }
)
def bronze_parsed_documents():
    return spark.readStream.table("healthcare_catalog.raw_zone.parsed_documents")

# SILVER TABLE: Extracted entities
@dlt.table(
    name="silver_extracted_entities",
    comment="Clinical entities extracted using DBRX LLM",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("has_entities", "extracted_entities IS NOT NULL")
@dlt.expect_or_drop("no_extraction_errors", "extracted_entities NOT LIKE '%error%'")
def silver_extracted_entities():
    return (
        dlt.read_stream("bronze_parsed_documents")
        .filter("parse_quality_score > 0.7")  # Only process high-quality parses
        .withColumn("extracted_entities", extract_entities_udf(F.col("text")))
        .withColumn("extraction_timestamp", F.current_timestamp())
        .select(
            "document_id",
            "path",
            "text",
            "sections",
            "extracted_entities",
            "extraction_timestamp"
        )
    )