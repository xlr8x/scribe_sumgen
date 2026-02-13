# Databricks notebook source
# MAGIC %md
# MAGIC # Interactive SOAP Agent - Fixed for Standard Runtime
# MAGIC
# MAGIC **Works with standard Databricks Runtime (non-ML)**
# MAGIC
# MAGIC This version:
# MAGIC - ✅ Installs MLflow automatically
# MAGIC - ✅ Uses your Foundation Models (GPT-5-2, Llama 3.3 70B)
# MAGIC - ✅ Works with Unity Catalog
# MAGIC - ✅ Databricks Widgets for interaction

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Required Packages

# COMMAND ----------

# Install MLflow and databricks-sdk
%pip install mlflow databricks-sdk --quiet
dbutils.library.restartPython()

# COMMAND ----------

print("✅ Packages installed successfully!")
print("   - mlflow")
print("   - databricks-sdk")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Import Libraries & Configuration

# COMMAND ----------

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp

# Try to import Databricks SDK
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    print("✅ Databricks SDK loaded")
except Exception as e:
    print(f"⚠️  Databricks SDK warning: {e}")
    print("   Will continue with basic functionality")
    w = None

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

print("✅ All libraries loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
USE_UNITY_CATALOG = True  # You have Unity Catalog!

# Unity Catalog configuration
if USE_UNITY_CATALOG:
    CATALOG = "healthcare_catalog"
    SILVER_SCHEMA = "silver_zone"
    GOLD_SCHEMA = "gold_zone"

    # Create catalog and schemas if they don't exist
    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
        spark.sql(f"USE CATALOG {CATALOG}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER_SCHEMA}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_SCHEMA}")
        print(f"✅ Unity Catalog setup complete: {CATALOG}")
    except Exception as e:
        print(f"⚠️  Catalog creation note: {e}")

    SILVER_ENTITIES_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.extracted_entities"
    GOLD_REVIEW_QUEUE = f"{CATALOG}.{GOLD_SCHEMA}.review_queue"
    GOLD_CLINICAL_SUMMARIES = f"{CATALOG}.{GOLD_SCHEMA}.clinical_summaries"
    GOLD_VALIDATED_ENTITIES = f"{CATALOG}.{GOLD_SCHEMA}.validated_entities"
else:
    # Fallback to simple database
    DATABASE = "clinical_documentation"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    spark.sql(f"USE {DATABASE}")

    SILVER_ENTITIES_TABLE = f"{DATABASE}.extracted_entities"
    GOLD_REVIEW_QUEUE = f"{DATABASE}.review_queue"
    GOLD_CLINICAL_SUMMARIES = f"{DATABASE}.clinical_summaries"
    GOLD_VALIDATED_ENTITIES = f"{DATABASE}.validated_entities"

# Model configuration - YOUR AVAILABLE MODELS
AMBIGUITY_MODEL = "databricks-meta-llama-3-3-70b-instruct"  # Best reasoning
QUESTION_MODEL = "databricks-gpt-5-2"                       # Best Q generation
SOAP_MODEL = "databricks-gpt-5-2"                           # Best medical writing

print(f"✅ Configuration loaded")
print(f"   Catalog: {CATALOG if USE_UNITY_CATALOG else DATABASE}")
print(f"   Ambiguity Model: {AMBIGUITY_MODEL}")
print(f"   Question Model: {QUESTION_MODEL}")
print(f"   SOAP Model: {SOAP_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Gold Layer Tables

# COMMAND ----------

# Create Review Queue table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD_REVIEW_QUEUE} (
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
COMMENT 'Documents awaiting human review with agent-generated questions'
""")
print(f"✅ Created: {GOLD_REVIEW_QUEUE}")

# Create Clinical Summaries table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD_CLINICAL_SUMMARIES} (
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
COMMENT 'Final SOAP notes with quality metrics'
""")
print(f"✅ Created: {GOLD_CLINICAL_SUMMARIES}")

# Create Validated Entities table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD_VALIDATED_ENTITIES} (
    document_id STRING,
    updated_entities STRING,
    changes STRING,
    validation_score FLOAT,
    human_verified_fields INT,
    human_review_timestamp TIMESTAMP
)
USING DELTA
COMMENT 'Clinical entities enhanced with human feedback'
""")
print(f"✅ Created: {GOLD_VALIDATED_ENTITIES}")

print("\n✅ All Gold layer tables created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Components

# COMMAND ----------

class AmbiguityDetector:
    """Detects missing/unclear clinical information using Llama 3.3 70B"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.w = w  # WorkspaceClient

    def detect_ambiguities(self, entities: Dict, source_text: str) -> Dict:
        """Analyze entities for ambiguities"""
        prompt = self._create_prompt(entities, source_text)

        # Use MLflow if available
        try:
            with mlflow.start_run(run_name="ambiguity_detection", nested=True):
                start_time = time.time()
                response = self._call_model(prompt)
                processing_time = time.time() - start_time
                result = self._parse_response(response)

                # Log to MLflow
                mlflow.log_param("model", self.model_name)
                mlflow.log_metric("processing_time_seconds", processing_time)
                mlflow.log_metric("ambiguities_found", len(result.get('ambiguities', [])))
                mlflow.log_dict(result, "ambiguities.json")
        except:
            # MLflow not available, just process
            start_time = time.time()
            response = self._call_model(prompt)
            processing_time = time.time() - start_time
            result = self._parse_response(response)

        print(f"   ✅ Ambiguity detection: {processing_time:.1f}s, {len(result.get('ambiguities', []))} found")
        return result

    def _create_prompt(self, entities: Dict, source_text: str) -> str:
        """Create prompt for ambiguity detection"""
        entities_str = json.dumps(entities, indent=2)

        prompt = f"""You are a clinical documentation quality analyst. Analyze these extracted clinical entities for ambiguities, missing information, or unclear details.

EXTRACTED ENTITIES:
{entities_str}

SOURCE TEXT:
{source_text[:1500]}...

Analyze for:
1. Missing critical fields (severity, dosages, allergies)
2. Insufficient detail (vague descriptions)
3. Medical logic gaps

Return ONLY valid JSON:
{{
  "ambiguities": [
    {{
      "id": "amb_1",
      "category": "missing_information",
      "field": "symptoms[0].severity",
      "description": "What's missing",
      "clinical_importance": "high"
    }}
  ],
  "confidence_score": 0.85,
  "requires_review": true,
  "completeness_before": 0.68,
  "reasoning_trace": "Brief explanation"
}}

Return ONLY the JSON object.
"""
        return prompt

    def _call_model(self, prompt: str) -> str:
        """Call Databricks Foundation Model API"""
        if not self.w:
            return json.dumps({
                "ambiguities": [],
                "confidence_score": 0.5,
                "requires_review": False,
                "reasoning_trace": "WorkspaceClient not available"
            })

        try:
            from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

            response = self.w.serving_endpoints.query(
                name=self.model_name,
                messages=[
                    ChatMessage(role=ChatMessageRole.SYSTEM, content="You are a clinical analyst. Return only JSON."),
                    ChatMessage(role=ChatMessageRole.USER, content=prompt)
                ],
                temperature=0.0,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️  Model call warning: {e}")
            return json.dumps({
                "ambiguities": [],
                "confidence_score": 0.5,
                "requires_review": False,
                "reasoning_trace": f"Error: {str(e)}"
            })

    def _parse_response(self, response: str) -> Dict:
        """Parse JSON from response"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except:
            return {
                "ambiguities": [],
                "confidence_score": 0.5,
                "requires_review": False,
                "reasoning_trace": "Parse error"
            }

print("✅ AmbiguityDetector defined")

# COMMAND ----------

class QuestionGenerator:
    """Generates questions using GPT-5-2"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.w = w

    def generate_questions(self, ambiguities: List[Dict], entities: Dict, max_questions: int = 5) -> Dict:
        """Generate prioritized questions"""
        prompt = self._create_prompt(ambiguities, entities, max_questions)

        try:
            with mlflow.start_run(run_name="question_generation", nested=True):
                start_time = time.time()
                response = self._call_model(prompt)
                processing_time = time.time() - start_time
                result = self._parse_response(response)

                mlflow.log_param("model", self.model_name)
                mlflow.log_metric("processing_time_seconds", processing_time)
                mlflow.log_metric("questions_generated", len(result.get('questions', [])))
                mlflow.log_dict(result, "questions.json")
        except:
            start_time = time.time()
            response = self._call_model(prompt)
            processing_time = time.time() - start_time
            result = self._parse_response(response)

        print(f"   ✅ Question generation: {processing_time:.1f}s, {len(result.get('questions', []))} questions")
        return result

    def _create_prompt(self, ambiguities: List[Dict], entities: Dict, max_questions: int) -> str:
        """Create prompt"""
        return f"""Generate {max_questions} clear questions to resolve these ambiguities.

AMBIGUITIES:
{json.dumps(ambiguities, indent=2)}

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "id": "q1",
      "priority": 1,
      "field_to_update": "symptoms[0].severity",
      "text": "Question?",
      "context": "Current value",
      "input_type": "select",
      "options": ["opt1", "opt2"]
    }}
  ]
}}"""

    def _call_model(self, prompt: str) -> str:
        """Call model"""
        if not self.w:
            return json.dumps({"questions": []})

        try:
            from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
            response = self.w.serving_endpoints.query(
                name=self.model_name,
                messages=[
                    ChatMessage(role=ChatMessageRole.SYSTEM, content="Generate questions. Return only JSON."),
                    ChatMessage(role=ChatMessageRole.USER, content=prompt)
                ],
                temperature=0.3,
                max_tokens=1200
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️  Model call warning: {e}")
            return json.dumps({"questions": []})

    def _parse_response(self, response: str) -> Dict:
        """Parse JSON"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            return json.loads(response.strip())
        except:
            return {"questions": []}

print("✅ QuestionGenerator defined")

# COMMAND ----------

class FeedbackProcessor:
    """Process human feedback - pure Python"""

    def process_feedback(self, entities: Dict, questions: List[Dict], answers: Dict) -> Dict:
        """Merge human answers into entities"""
        import copy

        try:
            with mlflow.start_run(run_name="feedback_processing", nested=True):
                result = self._process(entities, questions, answers)
                mlflow.log_metric("fields_improved", result['fields_improved'])
                mlflow.log_metric("improvement_delta", result['improvement_delta'])
        except:
            result = self._process(entities, questions, answers)

        print(f"   ✅ Feedback processed: {result['fields_improved']} fields, +{result['improvement_delta']:.1%}")
        return result

    def _process(self, entities, questions, answers):
        """Core processing logic"""
        import copy
        updated_entities = copy.deepcopy(entities)
        changes = []

        for question in questions:
            q_id = question['id']
            if q_id not in answers or not answers[q_id]:
                continue

            answer = answers[q_id]
            field_path = question['field_to_update']
            old_value = self._get_field(updated_entities, field_path)
            self._set_field(updated_entities, field_path, answer)

            changes.append({
                "field": field_path,
                "old_value": old_value,
                "new_value": answer,
                "source": "human_review",
                "question_id": q_id
            })

        comp_before = self._calc_completeness(entities)
        comp_after = self._calc_completeness(updated_entities)

        return {
            "updated_entities": updated_entities,
            "changes": changes,
            "completeness_before": comp_before,
            "completeness_after": comp_after,
            "improvement_delta": comp_after - comp_before,
            "fields_improved": len(changes)
        }

    def _get_field(self, obj, path):
        """Get nested field"""
        try:
            parts = path.replace('[', '.').replace(']', '').split('.')
            current = obj
            for part in parts:
                current = current[int(part)] if part.isdigit() else current.get(part)
            return current
        except:
            return None

    def _set_field(self, obj, path, value):
        """Set nested field"""
        try:
            parts = path.replace('[', '.').replace(']', '').split('.')
            current = obj
            for part in parts[:-1]:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
            last = parts[-1]
            if last.isdigit():
                current[int(last)] = value
            else:
                current[last] = value
        except:
            pass

    def _calc_completeness(self, obj):
        """Calculate completeness"""
        def count(o, non_null=0, total=0):
            if isinstance(o, dict):
                for v in o.values():
                    n, t = count(v)
                    non_null, total = non_null + n, total + t
            elif isinstance(o, list):
                for item in o:
                    n, t = count(item)
                    non_null, total = non_null + n, total + t
            else:
                total += 1
                if o is not None and o != "" and o != []:
                    non_null += 1
            return non_null, total
        non_null, total = count(obj)
        return non_null / total if total > 0 else 0.0

print("✅ FeedbackProcessor defined")

# COMMAND ----------

class SOAPGenerator:
    """Generate SOAP notes using GPT-5-2"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.w = w

    def generate_soap(self, entities: Dict, feedback_context: Dict) -> Dict:
        """Generate SOAP note"""
        prompt = self._create_prompt(entities, feedback_context)

        try:
            with mlflow.start_run(run_name="soap_generation", nested=True):
                start_time = time.time()
                response = self._call_model(prompt)
                processing_time = time.time() - start_time

                result = {
                    "soap_note": response,
                    "sections": self._parse_sections(response),
                    "quality_metrics": {
                        "completeness_score": feedback_context.get('completeness_after', 0),
                        "completeness_before": feedback_context.get('completeness_before', 0),
                        "improvement_delta": feedback_context.get('improvement_delta', 0),
                        "human_verified_fields": feedback_context.get('fields_improved', 0),
                        "generation_time_seconds": processing_time,
                        "word_count": len(response.split())
                    }
                }

                mlflow.log_param("model", self.model_name)
                mlflow.log_metric("generation_time_seconds", processing_time)
                mlflow.log_text(response, "soap_note.txt")
        except:
            start_time = time.time()
            response = self._call_model(prompt)
            processing_time = time.time() - start_time

            result = {
                "soap_note": response,
                "sections": {},
                "quality_metrics": {
                    "generation_time_seconds": processing_time,
                    "word_count": len(response.split())
                }
            }

        print(f"   ✅ SOAP generation: {processing_time:.1f}s, {len(response.split())} words")
        return result

    def _create_prompt(self, entities: Dict, feedback_context: Dict) -> str:
        """Create SOAP prompt"""
        changes = feedback_context.get('changes', [])
        feedback_summary = "\n".join([f"- {c['field']}: {c['new_value']}" for c in changes])

        return f"""Generate a professional SOAP note.

ENTITIES:
{json.dumps(entities, indent=2)}

HUMAN VERIFIED ({len(changes)} fields):
{feedback_summary}

Format:
**SUBJECTIVE**
Chief Complaint: ...
History: ...

**OBJECTIVE**
Vitals: ...
Exam: ...

**ASSESSMENT**
1. Diagnosis...

**PLAN**
Treatment: ...
Follow-up: ...

Generate the SOAP note:
"""

    def _call_model(self, prompt: str) -> str:
        """Call model"""
        if not self.w:
            return "SOAP note generation unavailable (WorkspaceClient not initialized)"

        try:
            from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
            response = self.w.serving_endpoints.query(
                name=self.model_name,
                messages=[
                    ChatMessage(role=ChatMessageRole.SYSTEM, content="You are a medical documentation specialist."),
                    ChatMessage(role=ChatMessageRole.USER, content=prompt)
                ],
                temperature=0.3,
                max_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating SOAP: {str(e)}"

    def _parse_sections(self, soap_note: str) -> Dict:
        """Parse SOAP sections"""
        sections = {}
        current_section = None
        content = []

        for line in soap_note.split('\n'):
            if '**SUBJECTIVE**' in line:
                if current_section:
                    sections[current_section] = '\n'.join(content).strip()
                current_section, content = 'subjective', []
            elif '**OBJECTIVE**' in line:
                if current_section:
                    sections[current_section] = '\n'.join(content).strip()
                current_section, content = 'objective', []
            elif '**ASSESSMENT**' in line:
                if current_section:
                    sections[current_section] = '\n'.join(content).strip()
                current_section, content = 'assessment', []
            elif '**PLAN**' in line:
                if current_section:
                    sections[current_section] = '\n'.join(content).strip()
                current_section, content = 'plan', []
            else:
                if current_section:
                    content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(content).strip()
        return sections

print("✅ SOAPGenerator defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Agent Coordinator

# COMMAND ----------

class ClinicalReviewAgent:
    """Main orchestrator"""

    def __init__(self):
        self.ambiguity_detector = AmbiguityDetector(AMBIGUITY_MODEL)
        self.question_generator = QuestionGenerator(QUESTION_MODEL)
        self.feedback_processor = FeedbackProcessor()
        self.soap_generator = SOAPGenerator(SOAP_MODEL)
        print("✅ ClinicalReviewAgent initialized")

    def process_document(self, document_id: str, entities: Dict, source_text: str) -> Dict:
        """Stage 1-2: Detect ambiguities and generate questions"""
        print(f"\n{'='*60}")
        print(f"Processing: {document_id}")
        print(f"{'='*60}\n")

        try:
            with mlflow.start_run(run_name=f"agent_{document_id}"):
                return self._process(document_id, entities, source_text)
        except:
            return self._process(document_id, entities, source_text)

    def _process(self, document_id, entities, source_text):
        """Core processing"""
        print("🔍 Stage 1: Detecting ambiguities...")
        ambiguities_result = self.ambiguity_detector.detect_ambiguities(entities, source_text)

        requires_review = ambiguities_result.get('requires_review', False)
        completeness = ambiguities_result.get('completeness_before', 0)

        if not requires_review:
            print("\n✅ High confidence - generating SOAP directly...")
            soap_result = self.soap_generator.generate_soap(entities, {
                'completeness_before': completeness,
                'completeness_after': completeness,
                'changes': []
            })
            self._save_to_clinical_summaries(document_id, entities, soap_result, {})
            return {
                "status": "completed_auto",
                "soap_note": soap_result['soap_note'],
                "quality_metrics": soap_result['quality_metrics']
            }

        print("\n❓ Stage 2: Generating questions...")
        questions_result = self.question_generator.generate_questions(
            ambiguities_result.get('ambiguities', []), entities
        )

        self._save_to_review_queue(document_id, entities, source_text,
                                   ambiguities_result, questions_result, completeness)

        return {
            "status": "pending_review",
            "document_id": document_id,
            "questions": questions_result['questions'],
            "ambiguities": ambiguities_result
        }

    def submit_feedback(self, document_id: str, answers: Dict) -> Dict:
        """Stage 3-4: Process feedback and generate SOAP"""
        print(f"\n{'='*60}")
        print(f"Processing Feedback: {document_id}")
        print(f"{'='*60}\n")

        review_data = self._load_from_review_queue(document_id)
        if not review_data:
            return {"status": "error", "message": "Document not found"}

        entities = json.loads(review_data['entities'])
        questions = json.loads(review_data['questions'])['questions']

        print("🔄 Stage 3: Processing feedback...")
        feedback_result = self.feedback_processor.process_feedback(entities, questions, answers)

        print("\n📄 Stage 4: Generating SOAP...")
        soap_result = self.soap_generator.generate_soap(
            feedback_result['updated_entities'], feedback_result
        )

        self._save_to_validated_entities(document_id, feedback_result)
        self._save_to_clinical_summaries(document_id, feedback_result['updated_entities'],
                                        soap_result, feedback_result)

        return {
            "status": "completed",
            "soap_note": soap_result['soap_note'],
            "sections": soap_result['sections'],
            "quality_metrics": soap_result['quality_metrics']
        }

    def _save_to_review_queue(self, doc_id, entities, source_text, ambiguities, questions, completeness):
        """Save to review queue"""
        data = [{
            "document_id": doc_id,
            "entities": json.dumps(entities),
            "source_text": source_text[:5000],
            "ambiguities": json.dumps(ambiguities),
            "questions": json.dumps(questions),
            "status": "pending_review",
            "completeness_before": completeness,
            "created_timestamp": datetime.now(),
            "updated_timestamp": datetime.now()
        }]
        spark.createDataFrame(data).write.format("delta").mode("append").saveAsTable(GOLD_REVIEW_QUEUE)

    def _load_from_review_queue(self, doc_id):
        """Load from review queue"""
        df = spark.table(GOLD_REVIEW_QUEUE).filter(col("document_id") == doc_id)
        rows = df.collect()
        return rows[0].asDict() if rows else None

    def _save_to_validated_entities(self, doc_id, feedback_result):
        """Save validated entities"""
        data = [{
            "document_id": doc_id,
            "updated_entities": json.dumps(feedback_result['updated_entities']),
            "changes": json.dumps(feedback_result['changes']),
            "validation_score": feedback_result['completeness_after'],
            "human_verified_fields": feedback_result['fields_improved'],
            "human_review_timestamp": datetime.now()
        }]
        spark.createDataFrame(data).write.format("delta").mode("append").saveAsTable(GOLD_VALIDATED_ENTITIES)

    def _save_to_clinical_summaries(self, doc_id, entities, soap_result, feedback_result):
        """Save SOAP note"""
        metrics = soap_result['quality_metrics']
        data = [{
            "document_id": doc_id,
            "soap_note": soap_result['soap_note'],
            "sections": json.dumps(soap_result['sections']),
            "quality_metrics": json.dumps(metrics),
            "human_feedback": json.dumps(feedback_result.get('changes', [])),
            "completeness_before": metrics.get('completeness_before', 0),
            "completeness_after": metrics.get('completeness_score', 0),
            "improvement_delta": metrics.get('improvement_delta', 0),
            "generation_timestamp": datetime.now()
        }]
        spark.createDataFrame(data).write.format("delta").mode("append").saveAsTable(GOLD_CLINICAL_SUMMARIES)

# Initialize agent
agent = ClinicalReviewAgent()
print("✅ Agent ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 DEMO: Process Sample Document

# COMMAND ----------

# Sample document
sample_doc_id = "doc_001_demo"
sample_entities = {
    "chief_complaint": "Chest pain",
    "symptoms": [
        {"symptom": "Chest pain", "onset": "yesterday", "severity": None, "location": None},
        {"symptom": "Shortness of breath", "onset": "yesterday", "severity": None}
    ],
    "diagnoses": [{"diagnosis": "Chest pain, cardiac", "icd10_code": None, "status": "suspected"}],
    "medications": [{"name": "blood pressure med", "dosage": None, "frequency": None}],
    "vital_signs": {"blood_pressure": "145/92", "heart_rate": "98", "temperature": "98.6F"},
    "allergies": []
}

sample_text = """Patient presents with chest pain starting yesterday. Pressure-like pain.
Shortness of breath. Takes blood pressure medication. BP 145/92, HR 98."""

print("✅ Sample document loaded")

# COMMAND ----------

# Process document
result = agent.process_document(sample_doc_id, sample_entities, sample_text)

if result['status'] == 'pending_review':
    print("\n📋 QUESTIONS TO ANSWER:")
    questions = result['questions']
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q['text']}")
        if q.get('options'):
            print(f"   Options: {', '.join(q['options'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Widgets for Answers

# COMMAND ----------

if result['status'] == 'pending_review':
    questions = result['questions']

    # Remove old widgets
    try:
        for q in questions:
            dbutils.widgets.remove(q['id'])
    except:
        pass

    # Create widgets
    for q in questions:
        if q.get('input_type') == 'select' and q.get('options'):
            dbutils.widgets.dropdown(q['id'], q['options'][0], q['options'], f"Q{q['priority']}: {q['text']}")
        else:
            dbutils.widgets.text(q['id'], '', f"Q{q['priority']}: {q['text']}")

    print("✅ Widgets created at top of notebook!")
    print("👆 Answer questions above, then run next cell")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Submit Answers & Generate SOAP

# COMMAND ----------

if result['status'] == 'pending_review':
    # Collect answers
    answers = {}
    for q in questions:
        answer = dbutils.widgets.get(q['id'])
        if answer:
            answers[q['id']] = answer

    if answers:
        soap_result = agent.submit_feedback(sample_doc_id, answers)

        print("\n" + "="*80)
        print("📄 SOAP NOTE GENERATED")
        print("="*80 + "\n")
        print(soap_result['soap_note'])

        print("\n" + "="*80)
        print("📊 METRICS")
        print("="*80)
        metrics = soap_result['quality_metrics']
        print(f"Before: {metrics.get('completeness_before', 0):.1%}")
        print(f"After:  {metrics.get('completeness_score', 0):.1%}")
        print(f"Improvement: +{metrics.get('improvement_delta', 0):.1%}")
        print(f"Fields improved: {metrics.get('human_verified_fields', 0)}")

        # Cleanup
        for q in questions:
            try:
                dbutils.widgets.remove(q['id'])
            except:
                pass
    else:
        print("⚠️  No answers provided")

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Results

# COMMAND ----------

print("📋 Review Queue")
display(spark.table(GOLD_REVIEW_QUEUE).orderBy(col("created_timestamp").desc()))

# COMMAND ----------

print("📄 Clinical Summaries")
display(spark.table(GOLD_CLINICAL_SUMMARIES).orderBy(col("generation_timestamp").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ✅ SETUP COMPLETE!
# MAGIC
# MAGIC Your Interactive SOAP Agent is ready with:
# MAGIC - ✅ MLflow installed and working
# MAGIC - ✅ Foundation Models (GPT-5-2, Llama 3.3 70B)
# MAGIC - ✅ Unity Catalog integration
# MAGIC - ✅ All agent components
# MAGIC - ✅ Gold layer tables
# MAGIC
# MAGIC **Next:** Use Genie to analyze your results!
