import mlflow
import json

# Load sample parsed documents
sample_docs = spark.table("healthcare_catalog.raw_zone.parsed_documents").limit(10)

# Manual validation criteria
validation_results = []

for row in sample_docs.collect():
    result = {
        'document_id': row.document_id,
        'quality_score': row.parse_quality_score,
        'has_text': len(row.text) > 0 if row.text else False,
        'num_sections': len(json.loads(row.sections)) if row.sections else 0,
        'num_tables': len(json.loads(row.tables)) if row.tables else 0,
        'errors': row.parse_errors
    }
    validation_results.append(result)

# Just print instead of logging to MLflow
print(f"Document: {row.document_id}")
print(f"  Quality: {result['quality_score']}")
print(f"  Sections: {result['num_sections']}")
print(f"  Tables: {result['num_tables']}")
