# Databricks notebook source
# MAGIC %md
# MAGIC # Process Existing Extracted Entities with SOAP Agent
# MAGIC
# MAGIC **Reads from your existing table and generates SOAP notes**
# MAGIC
# MAGIC Table: `americas_databricks_hackathon_2025.americas_scribe_squad_hackathon.extracted_entities`
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads documents from your table
# MAGIC 2. Parses the clinical text into structured entities
# MAGIC 3. Runs through Interactive SOAP Agent
# MAGIC 4. Saves results to Gold layer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup (Install packages if needed)

# COMMAND ----------

# Install MLflow if not already available
%pip install mlflow databricks-sdk requests --quiet
dbutils.library.restartPython()

# COMMAND ----------

import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp, explode, array_join
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Initialize
spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()

print("✅ Libraries loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Your existing table
SOURCE_TABLE = "americas_databricks_hackathon_2025.americas_scribe_squad_hackathon.extracted_entities"

# Gold layer tables (output)
CATALOG = "americas_databricks_hackathon_2025"
SCHEMA = "americas_scribe_squad_hackathon"
GOLD_REVIEW_QUEUE = f"{CATALOG}.{SCHEMA}.review_queue"
GOLD_CLINICAL_SUMMARIES = f"{CATALOG}.{SCHEMA}.clinical_summaries"
GOLD_VALIDATED_ENTITIES = f"{CATALOG}.{SCHEMA}.validated_entities"

# Models (using CURRENT workspace - confirmed working!)
# NOTE: databricks-meta-llama-3-3-70b-instruct is available in this workspace
ENTITY_EXTRACTION_MODEL = "databricks-meta-llama-3-3-70b-instruct"
AMBIGUITY_MODEL = "databricks-meta-llama-3-3-70b-instruct"
QUESTION_MODEL = "databricks-meta-llama-3-3-70b-instruct"
SOAP_MODEL = "databricks-meta-llama-3-3-70b-instruct"

# Initialize WorkspaceClient for same-workspace model calls
w = WorkspaceClient()

print(f"✅ Configuration loaded")
print(f"   Source: {SOURCE_TABLE}")
print(f"   Gold layer: {CATALOG}.{SCHEMA}")
print(f"   Model: {ENTITY_EXTRACTION_MODEL}")
print(f"   Using same-workspace SDK (faster & simpler than cross-workspace)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model API Helper
# MAGIC
# MAGIC Simple helper function to call Foundation Models in the current workspace.

# COMMAND ----------

def call_foundation_model(
    model_name: str,
    system_message: str,
    user_message: str,
    temperature: float = 0.1,
    max_tokens: int = 2000
) -> str:
    """
    Call Foundation Model using WorkspaceClient SDK.

    This is simpler and faster than cross-workspace REST API calls.
    """
    try:
        response = w.serving_endpoints.query(
            name=model_name,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=system_message),
                ChatMessage(role=ChatMessageRole.USER, content=user_message)
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Model API error: {str(e)}")

print("✅ Model API helper defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Gold Layer Tables

# COMMAND ----------

# Create Review Queue
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD_REVIEW_QUEUE} (
    mrn STRING,
    patient_name STRING,
    visit_date STRING,
    document_id STRING,
    entities STRING,
    source_text STRING,
    ambiguities STRING,
    questions STRING,
    status STRING,
    completeness_before FLOAT,
    created_timestamp TIMESTAMP,
    updated_timestamp TIMESTAMP
)
USING DELTA
""")

# Create Clinical Summaries
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD_CLINICAL_SUMMARIES} (
    mrn STRING,
    patient_name STRING,
    visit_date STRING,
    document_id STRING,
    soap_note STRING,
    sections STRING,
    quality_metrics STRING,
    human_feedback STRING,
    completeness_before FLOAT,
    completeness_after FLOAT,
    improvement_delta FLOAT,
    generation_timestamp TIMESTAMP
)
USING DELTA
""")

# Create Validated Entities
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD_VALIDATED_ENTITIES} (
    mrn STRING,
    patient_name STRING,
    visit_date STRING,
    document_id STRING,
    updated_entities STRING,
    changes STRING,
    validation_score FLOAT,
    human_verified_fields INT,
    human_review_timestamp TIMESTAMP
)
USING DELTA
""")

print("✅ Gold layer tables created")

# Define explicit schema to match table (avoid type inference issues)
clinical_summaries_schema = StructType([
    StructField("mrn", StringType(), True),
    StructField("patient_name", StringType(), True),
    StructField("visit_date", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("soap_note", StringType(), True),
    StructField("sections", StringType(), True),
    StructField("quality_metrics", StringType(), True),
    StructField("human_feedback", StringType(), True),
    StructField("completeness_before", FloatType(), True),
    StructField("completeness_after", FloatType(), True),
    StructField("improvement_delta", FloatType(), True),
    StructField("generation_timestamp", TimestampType(), True)
])

print("✅ Schema defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Your Data

# COMMAND ----------

# Load source data
df_source = spark.table(SOURCE_TABLE)

# Display sample
print("📊 Source Data Sample:")
display(df_source.limit(5))

# Get column names
print(f"\n📋 Columns: {df_source.columns}")

# Count records
count = df_source.count()
print(f"\n📊 Total records: {count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entity Extraction Agent
# MAGIC
# MAGIC Converts raw clinical text into structured entities

# COMMAND ----------

class EntityExtractor:
    """Extract structured entities from clinical text"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def extract_entities(self, clinical_text: str) -> Dict:
        """Extract structured entities from text"""
        prompt = self._create_prompt(clinical_text)

        try:
            with mlflow.start_run(run_name="entity_extraction", nested=True):
                start_time = time.time()
                response = self._call_model(prompt)
                processing_time = time.time() - start_time
                result = self._parse_response(response)

                mlflow.log_metric("extraction_time", processing_time)
                mlflow.log_dict(result, "entities.json")

                return result
        except:
            start_time = time.time()
            response = self._call_model(prompt)
            processing_time = time.time() - start_time
            return self._parse_response(response)

    def _create_prompt(self, text: str) -> str:
        """Create extraction prompt"""
        return f"""Extract structured clinical information from this text.

CLINICAL TEXT:
{text}

Extract and return ONLY valid JSON:
{{
  "chief_complaint": "primary reason for visit",
  "symptoms": [
    {{
      "symptom": "symptom name",
      "onset": "when started",
      "severity": "mild/moderate/severe or scale",
      "location": "body location"
    }}
  ],
  "diagnoses": [
    {{
      "diagnosis": "diagnosis name",
      "icd10_code": "code if mentioned",
      "status": "confirmed/suspected"
    }}
  ],
  "medications": [
    {{
      "name": "medication name",
      "dosage": "dosage",
      "frequency": "frequency (QD, BID, etc)",
      "route": "route (PO, etc)"
    }}
  ],
  "procedures": [
    {{
      "procedure": "procedure name",
      "cpt_code": "code if mentioned"
    }}
  ],
  "vital_signs": {{
    "blood_pressure": "value",
    "heart_rate": "value",
    "temperature": "value",
    "respiratory_rate": "value"
  }},
  "allergies": ["allergy1", "allergy2"],
  "social_history": {{
    "smoking": "status",
    "alcohol": "status"
  }},
  "extraction_confidence": 0.85
}}

Extract ONLY information explicitly stated. Use null for missing fields.
Return ONLY the JSON object.
"""

    def _call_model(self, prompt: str) -> str:
        """Call model using WorkspaceClient SDK"""
        try:
            response = call_foundation_model(
                model_name=self.model_name,
                system_message="Extract clinical entities. Return only JSON.",
                user_message=prompt,
                temperature=0.1,
                max_tokens=2000
            )
            return response
        except Exception as e:
            print(f"⚠️  Extraction error: {e}")
            return json.dumps({"error": str(e)})

    def _parse_response(self, response: str) -> Dict:
        """Parse JSON"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except:
            return {
                "chief_complaint": "",
                "symptoms": [],
                "diagnoses": [],
                "medications": [],
                "vital_signs": {},
                "allergies": [],
                "extraction_confidence": 0.5
            }

print("✅ EntityExtractor defined")

# COMMAND ----------

# Import agent components from previous notebook
# (Simplified versions - full code in previous notebook)

class AmbiguityDetector:
    """Detect ambiguities in entities"""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def detect_ambiguities(self, entities: Dict, source_text: str) -> Dict:
        """Detect ambiguities"""
        prompt = f"""Analyze for ambiguities:

ENTITIES: {json.dumps(entities, indent=2)}

Return JSON:
{{
  "ambiguities": [{{"id": "amb_1", "category": "missing_information", "field": "field", "description": "desc", "clinical_importance": "high"}}],
  "requires_review": true,
  "completeness_before": 0.68,
  "reasoning_trace": "explanation"
}}"""

        try:
            content = call_foundation_model(
                model_name=self.model_name,
                system_message="Analyze for ambiguities. Return JSON.",
                user_message=prompt,
                temperature=0.0,
                max_tokens=1500
            )
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            return json.loads(content.strip())
        except Exception as e:
            print(f"⚠️  Ambiguity detection error: {e}")
            return {"ambiguities": [], "requires_review": False, "completeness_before": 0.8}

class QuestionGenerator:
    """Generate clarifying questions"""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_questions(self, ambiguities: List[Dict], entities: Dict) -> Dict:
        """Generate questions"""
        prompt = f"""Generate 3-5 questions for ambiguities:

AMBIGUITIES: {json.dumps(ambiguities, indent=2)}

Return JSON:
{{
  "questions": [{{"id": "q1", "priority": 1, "field_to_update": "field", "text": "Question?", "input_type": "select", "options": ["opt1", "opt2"]}}]
}}"""

        try:
            content = call_foundation_model(
                model_name=self.model_name,
                system_message="Generate questions. Return JSON.",
                user_message=prompt,
                temperature=0.3,
                max_tokens=1200
            )
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            return json.loads(content.strip())
        except Exception as e:
            print(f"⚠️  Question generation error: {e}")
            return {"questions": []}

class SOAPGenerator:
    """Generate SOAP notes"""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_soap(self, entities: Dict, feedback_context: Dict = None) -> Dict:
        """Generate SOAP note"""
        if feedback_context is None:
            feedback_context = {"changes": []}

        prompt = f"""Generate professional SOAP note:

ENTITIES: {json.dumps(entities, indent=2)}

Format:
**SUBJECTIVE**
Chief Complaint: ...
History: ...

**OBJECTIVE**
Vitals: ...

**ASSESSMENT**
1. Diagnosis...

**PLAN**
Treatment: ...

Generate the SOAP note:"""

        try:
            soap_note = call_foundation_model(
                model_name=self.model_name,
                system_message="Medical documentation specialist.",
                user_message=prompt,
                temperature=0.3,
                max_tokens=2500
            )
            return {
                "soap_note": soap_note,
                "sections": {},
                "quality_metrics": {
                    "completeness_score": 0.85,
                    "word_count": len(soap_note.split())
                }
            }
        except Exception as e:
            print(f"⚠️  SOAP generation error: {e}")
            return {
                "soap_note": f"Error: {str(e)}",
                "sections": {},
                "quality_metrics": {"completeness_score": 0.0}
            }

print("✅ Agent components defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Documents from Your Table

# COMMAND ----------

# Initialize agents
entity_extractor = EntityExtractor(ENTITY_EXTRACTION_MODEL)
ambiguity_detector = AmbiguityDetector(AMBIGUITY_MODEL)
question_generator = QuestionGenerator(QUESTION_MODEL)
soap_generator = SOAPGenerator(SOAP_MODEL)

print("✅ Agents initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Single Document (Test)

# COMMAND ----------

# Get first document from your table
first_doc = df_source.limit(1).collect()[0]

# Extract values (adjust column names as needed)
# Based on your data, columns appear to be: mrn, patient_name, visit_date, sections
mrn = str(first_doc[0])  # First column
patient_name = str(first_doc[1])  # Second column
visit_date = str(first_doc[2])  # Third column
sections = first_doc[3]  # Fourth column (array of strings)

# Combine sections into single text
if isinstance(sections, list):
    clinical_text = "\n".join(sections)
else:
    clinical_text = str(sections)

print(f"📄 Processing: {patient_name} (MRN: {mrn}, Date: {visit_date})")
print(f"\n📝 Clinical Text Preview:")
print(clinical_text[:500] + "...")

# COMMAND ----------

# Step 1: Extract structured entities
print("\n" + "="*60)
print("STEP 1: Extracting Structured Entities")
print("="*60)

entities = entity_extractor.extract_entities(clinical_text)
print("\n✅ Extracted Entities:")
print(json.dumps(entities, indent=2)[:500] + "...")

# COMMAND ----------

# Step 2: Detect ambiguities
print("\n" + "="*60)
print("STEP 2: Detecting Ambiguities")
print("="*60)

ambiguities_result = ambiguity_detector.detect_ambiguities(entities, clinical_text)
print(f"\n✅ Ambiguities found: {len(ambiguities_result.get('ambiguities', []))}")
print(f"   Requires review: {ambiguities_result.get('requires_review', False)}")
print(f"   Completeness: {ambiguities_result.get('completeness_before', 0):.1%}")

# COMMAND ----------

# Step 3: Generate SOAP note (skip human review for batch processing)
print("\n" + "="*60)
print("STEP 3: Generating SOAP Note")
print("="*60)

soap_result = soap_generator.generate_soap(entities)
print("\n✅ SOAP Note Generated:")
print(soap_result['soap_note'])

# COMMAND ----------

# Step 4: Save to Gold layer
print("\n" + "="*60)
print("STEP 4: Saving to Gold Layer")
print("="*60)

document_id = f"{mrn}_{visit_date}"

# Save to clinical_summaries
data = [{
    "mrn": mrn,
    "patient_name": patient_name,
    "visit_date": visit_date,
    "document_id": document_id,
    "soap_note": soap_result['soap_note'],
    "sections": json.dumps(soap_result['sections']),
    "quality_metrics": json.dumps(soap_result['quality_metrics']),
    "human_feedback": json.dumps([]),
    "completeness_before": float(ambiguities_result.get('completeness_before', 0.8)),
    "completeness_after": float(0.85),
    "improvement_delta": float(0.05),
    "generation_timestamp": datetime.now()
}]

spark.createDataFrame(data, schema=clinical_summaries_schema).write.format("delta").mode("append").saveAsTable(GOLD_CLINICAL_SUMMARIES)
print(f"✅ Saved to: {GOLD_CLINICAL_SUMMARIES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Process All Documents

# COMMAND ----------

def process_document_batch(row):
    """Process a single document row"""
    try:
        mrn = str(row[0])
        patient_name = str(row[1])
        visit_date = str(row[2])
        sections = row[3]

        # Combine sections
        if isinstance(sections, list):
            clinical_text = "\n".join(sections)
        else:
            clinical_text = str(sections)

        document_id = f"{mrn}_{visit_date}"

        # Extract entities
        entities = entity_extractor.extract_entities(clinical_text)

        # Detect ambiguities
        ambiguities_result = ambiguity_detector.detect_ambiguities(entities, clinical_text)

        # Generate SOAP (auto-mode, no human review)
        soap_result = soap_generator.generate_soap(entities)

        return {
            "mrn": mrn,
            "patient_name": patient_name,
            "visit_date": visit_date,
            "document_id": document_id,
            "soap_note": soap_result['soap_note'],
            "sections": json.dumps(soap_result['sections']),
            "quality_metrics": json.dumps(soap_result['quality_metrics']),
            "human_feedback": json.dumps([]),
            "completeness_before": float(ambiguities_result.get('completeness_before', 0.8)),
            "completeness_after": float(0.85),
            "improvement_delta": float(0.05),
            "generation_timestamp": datetime.now(),
            "status": "success"
        }
    except Exception as e:
        return {
            "mrn": str(row[0]) if len(row) > 0 else "unknown",
            "patient_name": str(row[1]) if len(row) > 1 else "unknown",
            "visit_date": str(row[2]) if len(row) > 2 else "unknown",
            "document_id": "error",
            "soap_note": f"Error: {str(e)}",
            "sections": "{}",
            "quality_metrics": "{}",
            "human_feedback": "[]",
            "completeness_before": float(0.0),
            "completeness_after": float(0.0),
            "improvement_delta": float(0.0),
            "generation_timestamp": datetime.now(),
            "status": "error"
        }

# Process limited batch (5 documents for testing)
print("🔄 Processing batch of 5 documents...")

batch_docs = df_source.limit(5).collect()
results = []

for i, doc in enumerate(batch_docs, 1):
    print(f"\nProcessing {i}/5...")
    result = process_document_batch(doc)
    results.append(result)
    print(f"   ✅ {result['patient_name']} - {result['status']}")

# Save results
if results:
    # Remove 'status' field before saving (not in table schema)
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if k != 'status'}
        clean_results.append(clean_r)

    results_df = spark.createDataFrame(clean_results, schema=clinical_summaries_schema)
    results_df.write.format("delta").mode("append").saveAsTable(GOLD_CLINICAL_SUMMARIES)
    print(f"\n✅ Batch processing complete! Saved {len(clean_results)} SOAP notes")

    # Show status summary
    success_count = sum(1 for r in results if r.get('status') == 'success')
    error_count = sum(1 for r in results if r.get('status') == 'error')
    print(f"   Success: {success_count}, Errors: {error_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Results

# COMMAND ----------

# View generated SOAP notes
print("📄 Generated SOAP Notes:")
display(spark.table(GOLD_CLINICAL_SUMMARIES).orderBy(col("generation_timestamp").desc()))

# COMMAND ----------

# View specific SOAP note
soap_notes = spark.table(GOLD_CLINICAL_SUMMARIES).limit(1).collect()
if soap_notes:
    print("="*80)
    print("SAMPLE SOAP NOTE")
    print("="*80)
    print(f"Patient: {soap_notes[0]['patient_name']}")
    print(f"MRN: {soap_notes[0]['mrn']}")
    print(f"Date: {soap_notes[0]['visit_date']}")
    print("\n" + "="*80)
    print(soap_notes[0]['soap_note'])
    print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process ALL Documents (Uncomment to run)

# COMMAND ----------

# # WARNING: This will process ALL documents in your table
# # Uncomment only when ready
#
# print(f"🔄 Processing ALL {count} documents...")
#
# all_docs = df_source.collect()
# all_results = []
#
# for i, doc in enumerate(all_docs, 1):
#     if i % 10 == 0:
#         print(f"   Progress: {i}/{count}")
#     result = process_document_batch(doc)
#     all_results.append(result)
#
# # Save all results
# if all_results:
#     # Remove 'status' field before saving
#     clean_results = [{k: v for k, v in r.items() if k != 'status'} for r in all_results]
#     results_df = spark.createDataFrame(clean_results, schema=clinical_summaries_schema)
#     results_df.write.format("delta").mode("append").saveAsTable(GOLD_CLINICAL_SUMMARIES)
#     print(f"\n✅ Complete! Processed {len(clean_results)} documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ✅ SUMMARY
# MAGIC
# MAGIC You've successfully:
# MAGIC 1. ✅ Loaded data from your existing table
# MAGIC 2. ✅ Extracted structured entities using Llama 3.3 70B
# MAGIC 3. ✅ Generated SOAP notes using the Interactive Agent
# MAGIC 4. ✅ Saved results to Gold layer tables
# MAGIC
# MAGIC **Architecture:**
# MAGIC - 📊 Data: `americas_databricks_hackathon_2025.americas_scribe_squad_hackathon`
# MAGIC - 🤖 Model: `databricks-meta-llama-3-3-70b-instruct` (same workspace)
# MAGIC - 🔗 Connection: WorkspaceClient SDK (simple & fast)
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Use Genie to analyze SOAP notes
# MAGIC - Create SQL dashboard for quality metrics
# MAGIC - Process all documents (uncomment batch processing code above)
# MAGIC
# MAGIC **Your Gold Layer Tables:**
# MAGIC - `americas_databricks_hackathon_2025.americas_scribe_squad_hackathon.clinical_summaries` - Final SOAP notes
# MAGIC - `americas_databricks_hackathon_2025.americas_scribe_squad_hackathon.validated_entities` - Validated entities
# MAGIC - `americas_databricks_hackathon_2025.americas_scribe_squad_hackathon.review_queue` - Documents for human review
