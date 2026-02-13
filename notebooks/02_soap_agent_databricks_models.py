# Databricks notebook source
# MAGIC %md
# MAGIC # Interactive SOAP Agent - Databricks Foundation Models
# MAGIC
# MAGIC **Optimized for Your Available Databricks Models**
# MAGIC
# MAGIC This version uses:
# MAGIC - ✅ Databricks GPT-5-2 (latest, most capable)
# MAGIC - ✅ Llama 3.3 70B (strong reasoning)
# MAGIC - ✅ Llama 3.1 405B (optional, for complex cases)
# MAGIC - ✅ No external APIs needed (all built-in)
# MAGIC - ✅ Databricks Widgets for interaction
# MAGIC
# MAGIC **Model Selection Strategy:**
# MAGIC - **Ambiguity Detection**: Llama 3.3 70B (excellent reasoning, cost-effective)
# MAGIC - **Question Generation**: GPT-5-2 (best at natural language generation)
# MAGIC - **SOAP Generation**: GPT-5-2 (professional medical writing)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Configuration

# COMMAND ----------

# Install/import required packages
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Initialize
spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()

print("✅ Libraries loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC **Models Available in Your Workspace:**
# MAGIC - `databricks-gpt-5-2` - Latest GPT model (best overall)
# MAGIC - `databricks-gpt-5-1` - Previous GPT version
# MAGIC - `databricks-meta-llama-3-3-70b-instruct` - Llama 3.3 70B (excellent reasoning)
# MAGIC - `databricks-meta-llama-3-1-405b-instruct` - Llama 3.1 405B (most powerful)
# MAGIC - `databricks-meta-llama-3-1-8b-instruct` - Llama 3.1 8B (fast, lightweight)
# MAGIC - `databricks-qwen3-next-80b-a3b-instruct` - Qwen 80B
# MAGIC - `databricks-llama-4-maverick` - Next-gen Llama

# COMMAND ----------

# Configuration
USE_UNITY_CATALOG = True  # Set to False if you don't have Unity Catalog

# Database/Catalog configuration
if USE_UNITY_CATALOG:
    CATALOG = "healthcare_catalog"
    SILVER_SCHEMA = "silver_zone"
    GOLD_SCHEMA = "gold_zone"

    # Create catalog and schemas if they don't exist
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER_SCHEMA}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_SCHEMA}")

    SILVER_ENTITIES_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.extracted_entities"
    GOLD_REVIEW_QUEUE = f"{CATALOG}.{GOLD_SCHEMA}.review_queue"
    GOLD_CLINICAL_SUMMARIES = f"{CATALOG}.{GOLD_SCHEMA}.clinical_summaries"
    GOLD_VALIDATED_ENTITIES = f"{CATALOG}.{GOLD_SCHEMA}.validated_entities"
else:
    # Use single database
    DATABASE = "clinical_documentation"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    spark.sql(f"USE {DATABASE}")

    SILVER_ENTITIES_TABLE = f"{DATABASE}.extracted_entities"
    GOLD_REVIEW_QUEUE = f"{DATABASE}.review_queue"
    GOLD_CLINICAL_SUMMARIES = f"{DATABASE}.clinical_summaries"
    GOLD_VALIDATED_ENTITIES = f"{DATABASE}.validated_entities"

# Model configuration - OPTIMIZED FOR YOUR WORKSPACE
# Strategy: Use best model for each task
AMBIGUITY_MODEL = "databricks-meta-llama-3-3-70b-instruct"  # Excellent reasoning, faster
QUESTION_MODEL = "databricks-gpt-5-2"                       # Best natural language
SOAP_MODEL = "databricks-gpt-5-2"                           # Best medical writing

# Alternative configurations (uncomment to try):
# For maximum quality (higher cost):
# AMBIGUITY_MODEL = "databricks-meta-llama-3-1-405b-instruct"
# QUESTION_MODEL = "databricks-gpt-5-2"
# SOAP_MODEL = "databricks-meta-llama-3-1-405b-instruct"

# For speed/cost optimization (lower quality):
# AMBIGUITY_MODEL = "databricks-meta-llama-3-1-8b-instruct"
# QUESTION_MODEL = "databricks-gpt-5-1"
# SOAP_MODEL = "databricks-gpt-5-1"

print(f"✅ Configuration loaded")
print(f"   Catalog/Database: {CATALOG if USE_UNITY_CATALOG else DATABASE}")
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

# Create Clinical Summaries table (Final SOAP notes)
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

print("✅ Gold layer tables created successfully!")
print(f"   📋 {GOLD_REVIEW_QUEUE}")
print(f"   📄 {GOLD_CLINICAL_SUMMARIES}")
print(f"   ✅ {GOLD_VALIDATED_ENTITIES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Components

# COMMAND ----------

class AmbiguityDetector:
    """
    Detects missing, unclear, or contradictory information in extracted clinical entities.
    Uses Llama 3.3 70B for excellent reasoning at good speed.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.w = WorkspaceClient()

    def detect_ambiguities(self, entities: Dict, source_text: str) -> Dict:
        """
        Analyze entities for ambiguities and missing information.
        """
        prompt = self._create_prompt(entities, source_text)

        with mlflow.start_run(run_name="ambiguity_detection", nested=True):
            start_time = time.time()

            # Call Databricks Foundation Model API
            response = self._call_model(prompt)
            processing_time = time.time() - start_time

            # Parse response
            result = self._parse_response(response)

            # Log to MLflow
            mlflow.log_param("model", self.model_name)
            mlflow.log_metric("processing_time_seconds", processing_time)
            mlflow.log_metric("ambiguities_found", len(result.get('ambiguities', [])))
            mlflow.log_metric("confidence_score", result.get('confidence_score', 0))
            mlflow.log_dict(result, "ambiguities.json")

            print(f"   ✅ Ambiguity detection completed in {processing_time:.1f}s")
            print(f"      Model: {self.model_name}")
            print(f"      Ambiguities found: {len(result.get('ambiguities', []))}")

            return result

    def _create_prompt(self, entities: Dict, source_text: str) -> str:
        """Create prompt for ambiguity detection"""
        entities_str = json.dumps(entities, indent=2)

        prompt = f"""You are a clinical documentation quality analyst. Analyze these extracted clinical entities for ambiguities, missing information, or unclear details.

EXTRACTED ENTITIES:
{entities_str}

SOURCE TEXT (for reference):
{source_text[:1500]}...

Analyze for:
1. **Missing Critical Fields**: Chief complaint, diagnoses, medications with missing details
2. **Insufficient Detail**: Symptoms without severity/location, medications without dosages
3. **Medical Logic Gaps**: Medications without matching diagnoses, contradictions
4. **Temporal Inconsistencies**: Unclear onset times, conflicting dates

Return ONLY valid JSON in this exact format:
{{
  "ambiguities": [
    {{
      "id": "amb_1",
      "category": "missing_information|insufficient_detail|logic_gap|temporal_inconsistency",
      "field": "path.to.field (e.g., symptoms[0].severity)",
      "description": "Clear description of what's missing or unclear",
      "clinical_importance": "high|medium|low",
      "clinical_reasoning": "Why this matters for patient care"
    }}
  ],
  "confidence_score": 0.85,
  "requires_review": true,
  "completeness_before": 0.68,
  "critical_fields_missing": 5,
  "reasoning_trace": "Brief explanation of analysis"
}}

Return ONLY the JSON object, no other text.
"""
        return prompt

    def _call_model(self, prompt: str) -> str:
        """Call Databricks Foundation Model API"""
        try:
            response = self.w.serving_endpoints.query(
                name=self.model_name,
                messages=[
                    ChatMessage(
                        role=ChatMessageRole.SYSTEM,
                        content="You are a clinical documentation quality analyst. Return only valid JSON."
                    ),
                    ChatMessage(
                        role=ChatMessageRole.USER,
                        content=prompt
                    )
                ],
                temperature=0.0,  # Deterministic for analysis
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error calling model {self.model_name}: {e}")
            # Return default result on error
            return json.dumps({
                "ambiguities": [],
                "confidence_score": 0.5,
                "requires_review": False,
                "reasoning_trace": f"Error: {str(e)}"
            })

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response as JSON"""
        try:
            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            result = json.loads(response.strip())
            return result
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response: {response[:500]}...")
            # Return default structure
            return {
                "ambiguities": [],
                "confidence_score": 0.5,
                "requires_review": False,
                "reasoning_trace": "Failed to parse response"
            }

print("✅ AmbiguityDetector class defined")

# COMMAND ----------

class QuestionGenerator:
    """
    Generates clinically-relevant clarifying questions using GPT-5-2.
    GPT-5-2 excels at natural language generation and context understanding.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.w = WorkspaceClient()

    def generate_questions(self, ambiguities: List[Dict], entities: Dict, max_questions: int = 5) -> Dict:
        """Generate prioritized clarifying questions"""
        prompt = self._create_prompt(ambiguities, entities, max_questions)

        with mlflow.start_run(run_name="question_generation", nested=True):
            start_time = time.time()

            response = self._call_model(prompt)
            processing_time = time.time() - start_time

            result = self._parse_response(response)

            # Log to MLflow
            mlflow.log_param("model", self.model_name)
            mlflow.log_metric("processing_time_seconds", processing_time)
            mlflow.log_metric("questions_generated", len(result.get('questions', [])))
            mlflow.log_dict(result, "questions.json")

            print(f"   ✅ Question generation completed in {processing_time:.1f}s")
            print(f"      Model: {self.model_name}")
            print(f"      Questions generated: {len(result.get('questions', []))}")

            return result

    def _create_prompt(self, ambiguities: List[Dict], entities: Dict, max_questions: int) -> str:
        """Create prompt for question generation"""
        ambiguities_str = json.dumps(ambiguities, indent=2)
        entities_str = json.dumps(entities, indent=2)

        prompt = f"""Generate clear, clinically-relevant questions to resolve these ambiguities.

DETECTED AMBIGUITIES:
{ambiguities_str}

CURRENT ENTITIES (for context):
{entities_str}

Requirements:
- Prioritize by clinical_importance (high first)
- Generate maximum {max_questions} questions
- Use appropriate medical terminology
- Provide context from current entities
- Suggest answer options when applicable (multiple choice preferred)
- Make questions specific and actionable

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "id": "q1",
      "priority": 1,
      "ambiguity_id": "amb_1",
      "field_to_update": "symptoms[0].severity",
      "text": "Clear question text ending with ?",
      "context": "Current value or context from entities",
      "input_type": "select|multi_select|text|number",
      "options": ["option1", "option2", "option3"],
      "placeholder": "optional placeholder for text input",
      "clinical_note": "Why this matters clinically"
    }}
  ],
  "total_questions": 5,
  "estimated_review_time_seconds": 90
}}

Return ONLY the JSON object.
"""
        return prompt

    def _call_model(self, prompt: str) -> str:
        """Call Databricks Foundation Model API"""
        try:
            response = self.w.serving_endpoints.query(
                name=self.model_name,
                messages=[
                    ChatMessage(
                        role=ChatMessageRole.SYSTEM,
                        content="You are a clinical question generation specialist. Return only valid JSON."
                    ),
                    ChatMessage(
                        role=ChatMessageRole.USER,
                        content=prompt
                    )
                ],
                temperature=0.3,  # Slightly creative for natural questions
                max_tokens=1200
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error calling model {self.model_name}: {e}")
            return json.dumps({"questions": [], "total_questions": 0})

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response as JSON"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            result = json.loads(response.strip())
            return result
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            return {
                "questions": [],
                "total_questions": 0,
                "estimated_review_time_seconds": 0
            }

print("✅ QuestionGenerator class defined")

# COMMAND ----------

class FeedbackProcessor:
    """
    Processes human answers and merges them into entity structure.
    No LLM needed - pure Python logic for reliability and speed.
    """

    def process_feedback(self, entities: Dict, questions: List[Dict], answers: Dict) -> Dict:
        """Merge human feedback into entities"""
        with mlflow.start_run(run_name="feedback_processing", nested=True):
            start_time = time.time()

            # Deep copy entities to avoid mutation
            import copy
            updated_entities = copy.deepcopy(entities)
            changes = []

            # Process each answer
            for question in questions:
                q_id = question['id']
                if q_id not in answers or not answers[q_id]:
                    continue

                answer = answers[q_id]
                field_path = question['field_to_update']

                # Update entity
                old_value = self._get_field_value(updated_entities, field_path)
                self._set_field_value(updated_entities, field_path, answer)
                new_value = self._get_field_value(updated_entities, field_path)

                # Track change
                changes.append({
                    "field": field_path,
                    "old_value": old_value,
                    "new_value": new_value,
                    "source": "human_review",
                    "question_id": q_id
                })

            # Calculate metrics
            completeness_before = self._calculate_completeness(entities)
            completeness_after = self._calculate_completeness(updated_entities)

            result = {
                "updated_entities": updated_entities,
                "changes": changes,
                "completeness_before": completeness_before,
                "completeness_after": completeness_after,
                "improvement_delta": completeness_after - completeness_before,
                "fields_improved": len(changes),
                "human_review_timestamp": datetime.now().isoformat()
            }

            processing_time = time.time() - start_time

            # Log to MLflow
            mlflow.log_metric("processing_time_seconds", processing_time)
            mlflow.log_metric("fields_improved", len(changes))
            mlflow.log_metric("completeness_before", completeness_before)
            mlflow.log_metric("completeness_after", completeness_after)
            mlflow.log_metric("improvement_delta", completeness_after - completeness_before)
            mlflow.log_dict(result, "feedback.json")

            print(f"   ✅ Feedback processing completed in {processing_time:.1f}s")
            print(f"      Fields improved: {len(changes)}")
            print(f"      Completeness: {completeness_before:.1%} → {completeness_after:.1%} (+{completeness_after - completeness_before:.1%})")

            return result

    def _get_field_value(self, obj: Any, path: str) -> Any:
        """Get value from nested object using path like 'symptoms[0].severity'"""
        try:
            parts = path.replace('[', '.').replace(']', '').split('.')
            current = obj
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
            return current
        except:
            return None

    def _set_field_value(self, obj: Any, path: str, value: Any):
        """Set value in nested object using path"""
        try:
            parts = path.replace('[', '.').replace(']', '').split('.')
            current = obj
            for i, part in enumerate(parts[:-1]):
                if part.isdigit():
                    current = current[int(part)]
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

            last_part = parts[-1]
            if last_part.isdigit():
                current[int(last_part)] = value
            else:
                current[last_part] = value
        except Exception as e:
            print(f"❌ Error setting field {path}: {e}")

    def _calculate_completeness(self, entities: Dict) -> float:
        """Calculate percentage of non-null fields"""
        def count_fields(obj, non_null_count=0, total_count=0):
            if isinstance(obj, dict):
                for v in obj.values():
                    n, t = count_fields(v)
                    non_null_count += n
                    total_count += t
            elif isinstance(obj, list):
                for item in obj:
                    n, t = count_fields(item)
                    non_null_count += n
                    total_count += t
            else:
                total_count += 1
                if obj is not None and obj != "" and obj != []:
                    non_null_count += 1
            return non_null_count, total_count

        non_null, total = count_fields(entities)
        return non_null / total if total > 0 else 0.0

print("✅ FeedbackProcessor class defined")

# COMMAND ----------

class SOAPGenerator:
    """
    Generates professional SOAP notes using GPT-5-2.
    GPT-5-2 provides excellent medical writing with proper formatting.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.w = WorkspaceClient()

    def generate_soap(self, entities: Dict, feedback_context: Dict) -> Dict:
        """Generate SOAP note from validated entities"""
        prompt = self._create_prompt(entities, feedback_context)

        with mlflow.start_run(run_name="soap_generation", nested=True):
            start_time = time.time()

            response = self._call_model(prompt)
            processing_time = time.time() - start_time

            # Parse sections from response
            sections = self._parse_soap_sections(response)

            result = {
                "soap_note": response,
                "sections": sections,
                "quality_metrics": {
                    "completeness_score": feedback_context.get('completeness_after', 0),
                    "completeness_before": feedback_context.get('completeness_before', 0),
                    "improvement_delta": feedback_context.get('improvement_delta', 0),
                    "human_verified_fields": feedback_context.get('fields_improved', 0),
                    "generation_time_seconds": processing_time,
                    "word_count": len(response.split()),
                    "generation_timestamp": datetime.now().isoformat()
                },
                "reasoning_trace": f"Generated SOAP note with {feedback_context.get('fields_improved', 0)} human-verified fields"
            }

            # Log to MLflow
            mlflow.log_param("model", self.model_name)
            mlflow.log_metric("processing_time_seconds", processing_time)
            mlflow.log_metric("word_count", len(response.split()))
            mlflow.log_metric("human_verified_fields", feedback_context.get('fields_improved', 0))
            mlflow.log_text(response, "soap_note.txt")
            mlflow.log_dict(result, "soap_result.json")

            print(f"   ✅ SOAP generation completed in {processing_time:.1f}s")
            print(f"      Model: {self.model_name}")
            print(f"      Word count: {len(response.split())}")

            return result

    def _create_prompt(self, entities: Dict, feedback_context: Dict) -> str:
        """Create prompt for SOAP generation"""
        entities_str = json.dumps(entities, indent=2)

        # Create human feedback summary
        changes = feedback_context.get('changes', [])
        feedback_summary = "\n".join([
            f"- {change['field']}: {change['old_value']} → {change['new_value']}"
            for change in changes
        ])

        prompt = f"""Generate a professional SOAP note from these validated clinical entities.

VALIDATED CLINICAL DATA:
{entities_str}

HUMAN REVIEW CONTEXT:
The following fields were verified/enhanced by a clinical reviewer:
{feedback_summary}

Total fields improved: {len(changes)}

Generate a professional SOAP note in this format:

**SUBJECTIVE**

Chief Complaint: [primary reason for visit]

History of Present Illness:
[Detailed narrative of the present illness with all symptoms, onset, duration, severity, location, radiation, associated symptoms]

Past Medical History: [relevant PMH]

Medications: [current medications with dosages and frequencies]

Allergies: [drug allergies or NKDA]

Social History: [smoking, alcohol, relevant social factors]

**OBJECTIVE**

Vital Signs:
[All vital signs with units]

Physical Examination:
[Examination findings by system]

**ASSESSMENT**

1. [Primary diagnosis with ICD-10 if available] - [status: confirmed/suspected/differential]
   [Clinical reasoning for this diagnosis]

2. [Secondary diagnoses if applicable]

**PLAN**

Diagnostics:
[Ordered tests and studies]

Treatment:
[Medications, procedures, interventions]

Follow-up:
[Follow-up instructions, consultations, monitoring]

Requirements:
- Use professional medical terminology
- Be concise but complete
- Proper formatting and structure
- Include units for all measurements
- Note "Not documented" for genuinely missing information
- Emphasize human-verified details for clinical accuracy
- Add brief note at end mentioning number of human-verified fields

Generate the SOAP note now:
"""
        return prompt

    def _call_model(self, prompt: str) -> str:
        """Call Databricks Foundation Model API"""
        try:
            response = self.w.serving_endpoints.query(
                name=self.model_name,
                messages=[
                    ChatMessage(
                        role=ChatMessageRole.SYSTEM,
                        content="You are an expert medical documentation specialist. Generate professional SOAP notes."
                    ),
                    ChatMessage(
                        role=ChatMessageRole.USER,
                        content=prompt
                    )
                ],
                temperature=0.3,  # Balanced creativity for natural prose
                max_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error calling model {self.model_name}: {e}")
            return f"Error generating SOAP note: {str(e)}"

    def _parse_soap_sections(self, soap_note: str) -> Dict[str, str]:
        """Extract SOAP sections from generated note"""
        sections = {}
        current_section = None
        current_content = []

        for line in soap_note.split('\n'):
            if '**SUBJECTIVE**' in line:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'subjective'
                current_content = []
            elif '**OBJECTIVE**' in line:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'objective'
                current_content = []
            elif '**ASSESSMENT**' in line:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'assessment'
                current_content = []
            elif '**PLAN**' in line:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'plan'
                current_content = []
            else:
                if current_section:
                    current_content.append(line)

        # Add last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

print("✅ SOAPGenerator class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Agent Coordinator

# COMMAND ----------

class ClinicalReviewAgent:
    """
    Main orchestrator for the Interactive SOAP Agent workflow.
    Coordinates all 4 stages using Databricks Foundation Models.
    """

    def __init__(self, enable_human_review: bool = True):
        self.enable_human_review = enable_human_review

        # Initialize all agents with optimized models
        self.ambiguity_detector = AmbiguityDetector(AMBIGUITY_MODEL)
        self.question_generator = QuestionGenerator(QUESTION_MODEL)
        self.feedback_processor = FeedbackProcessor()
        self.soap_generator = SOAPGenerator(SOAP_MODEL)

        print("✅ ClinicalReviewAgent initialized")
        print(f"   Using Databricks Foundation Models:")
        print(f"   - Ambiguity: {AMBIGUITY_MODEL}")
        print(f"   - Questions: {QUESTION_MODEL}")
        print(f"   - SOAP: {SOAP_MODEL}")

    def process_document(self, document_id: str, entities: Dict, source_text: str) -> Dict:
        """
        Stage 1-2: Detect ambiguities and generate questions.
        """
        print(f"\n{'='*60}")
        print(f"Processing Document: {document_id}")
        print(f"{'='*60}\n")

        with mlflow.start_run(run_name=f"agent_workflow_{document_id}"):
            mlflow.log_param("document_id", document_id)
            mlflow.log_param("enable_human_review", self.enable_human_review)

            # Stage 1: Detect ambiguities
            print("🔍 Stage 1: Detecting ambiguities...")
            ambiguities_result = self.ambiguity_detector.detect_ambiguities(entities, source_text)

            requires_review = ambiguities_result.get('requires_review', False)
            completeness = ambiguities_result.get('completeness_before', 0)

            # If high confidence or human review disabled, skip to SOAP generation
            if not requires_review or not self.enable_human_review:
                print("\n✅ High confidence - Generating SOAP directly...")
                soap_result = self.soap_generator.generate_soap(entities, {
                    'completeness_before': completeness,
                    'completeness_after': completeness,
                    'improvement_delta': 0,
                    'fields_improved': 0,
                    'changes': []
                })

                # Save directly to clinical_summaries
                self._save_to_clinical_summaries(document_id, entities, soap_result, {})

                return {
                    "status": "completed_auto",
                    "document_id": document_id,
                    "soap_note": soap_result['soap_note'],
                    "quality_metrics": soap_result['quality_metrics']
                }

            # Stage 2: Generate questions
            print("\n❓ Stage 2: Generating clarifying questions...")
            questions_result = self.question_generator.generate_questions(
                ambiguities_result.get('ambiguities', []),
                entities
            )

            # Save to review queue
            self._save_to_review_queue(
                document_id, entities, source_text,
                ambiguities_result, questions_result, completeness
            )

            mlflow.log_metric("num_questions", len(questions_result.get('questions', [])))
            mlflow.log_metric("completeness", completeness)

            return {
                "status": "pending_review",
                "document_id": document_id,
                "questions": questions_result['questions'],
                "ambiguities": ambiguities_result,
                "estimated_review_time": questions_result.get('estimated_review_time_seconds', 0)
            }

    def submit_feedback(self, document_id: str, answers: Dict) -> Dict:
        """
        Stage 3-4: Process human feedback and generate SOAP note.
        """
        print(f"\n{'='*60}")
        print(f"Processing Feedback: {document_id}")
        print(f"{'='*60}\n")

        with mlflow.start_run(run_name=f"feedback_processing_{document_id}"):
            # Load from review queue
            review_data = self._load_from_review_queue(document_id)

            if not review_data:
                return {
                    "status": "error",
                    "message": f"Document {document_id} not found in review queue"
                }

            entities = json.loads(review_data['entities'])
            questions = json.loads(review_data['questions'])['questions']

            # Stage 3: Process feedback
            print("🔄 Stage 3: Processing human feedback...")
            feedback_result = self.feedback_processor.process_feedback(
                entities, questions, answers
            )

            # Stage 4: Generate SOAP note
            print("\n📄 Stage 4: Generating SOAP note...")
            soap_result = self.soap_generator.generate_soap(
                feedback_result['updated_entities'],
                feedback_result
            )

            # Save to Gold layer
            self._save_to_validated_entities(document_id, feedback_result)
            self._save_to_clinical_summaries(document_id, feedback_result['updated_entities'], soap_result, feedback_result)

            # Update review queue status
            self._update_review_queue_status(document_id, 'completed')

            mlflow.log_metric("improvement_delta", feedback_result['improvement_delta'])
            mlflow.log_metric("fields_improved", feedback_result['fields_improved'])

            return {
                "status": "completed",
                "document_id": document_id,
                "soap_note": soap_result['soap_note'],
                "sections": soap_result['sections'],
                "quality_metrics": {
                    **soap_result['quality_metrics'],
                    "completeness_improvement": f"+{feedback_result['improvement_delta']:.1%}",
                    "fields_improved": feedback_result['fields_improved']
                },
                "feedback_summary": feedback_result
            }

    def _save_to_review_queue(self, doc_id: str, entities: Dict, source_text: str,
                             ambiguities: Dict, questions: Dict, completeness: float):
        """Save document to review queue"""
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

        df = spark.createDataFrame(data)
        df.write.format("delta").mode("append").saveAsTable(GOLD_REVIEW_QUEUE)
        print(f"   ✅ Saved to review queue")

    def _load_from_review_queue(self, doc_id: str) -> Optional[Dict]:
        """Load document from review queue"""
        df = spark.table(GOLD_REVIEW_QUEUE).filter(col("document_id") == doc_id)
        rows = df.collect()
        return rows[0].asDict() if rows else None

    def _update_review_queue_status(self, doc_id: str, status: str):
        """Update status in review queue"""
        spark.sql(f"""
            UPDATE {GOLD_REVIEW_QUEUE}
            SET status = '{status}',
                updated_timestamp = current_timestamp()
            WHERE document_id = '{doc_id}'
        """)

    def _save_to_validated_entities(self, doc_id: str, feedback_result: Dict):
        """Save validated entities to Gold layer"""
        data = [{
            "document_id": doc_id,
            "updated_entities": json.dumps(feedback_result['updated_entities']),
            "changes": json.dumps(feedback_result['changes']),
            "validation_score": feedback_result['completeness_after'],
            "human_verified_fields": feedback_result['fields_improved'],
            "human_review_timestamp": datetime.now()
        }]

        df = spark.createDataFrame(data)
        df.write.format("delta").mode("append").saveAsTable(GOLD_VALIDATED_ENTITIES)
        print(f"   ✅ Saved to validated entities")

    def _save_to_clinical_summaries(self, doc_id: str, entities: Dict, soap_result: Dict, feedback_result: Dict):
        """Save final SOAP note to Gold layer"""
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

        df = spark.createDataFrame(data)
        df.write.format("delta").mode("append").saveAsTable(GOLD_CLINICAL_SUMMARIES)
        print(f"   ✅ Saved to clinical summaries")

# Initialize agent
agent = ClinicalReviewAgent(enable_human_review=True)

print("✅ Agent initialized and ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 INTERACTIVE DEMO
# MAGIC
# MAGIC ### Step 1: Load Sample Document

# COMMAND ----------

# Sample clinical document with intentional gaps
sample_doc_id = "doc_001_demo"
sample_entities = {
    "chief_complaint": "Chest pain",
    "symptoms": [
        {
            "symptom": "Chest pain",
            "onset": "yesterday evening",
            "severity": None,  # ❌ Missing
            "location": None,  # ❌ Missing
            "radiation": None  # ❌ Missing
        },
        {
            "symptom": "Shortness of breath",
            "onset": "yesterday evening",
            "severity": None,  # ❌ Missing
            "location": None
        },
        {
            "symptom": "Sweating",
            "onset": "yesterday evening",
            "severity": None,
            "location": None
        }
    ],
    "diagnoses": [
        {
            "diagnosis": "Chest pain, likely cardiac in origin",
            "icd10_code": None,  # ❌ Missing
            "status": "suspected"
        },
        {
            "diagnosis": "Rule out acute coronary syndrome",
            "icd10_code": None,  # ❌ Missing
            "status": "differential"
        }
    ],
    "medications": [
        {
            "name": "blood pressure medication",  # ❌ Vague
            "dosage": None,  # ❌ Missing
            "frequency": None,  # ❌ Missing
            "route": None  # ❌ Missing
        }
    ],
    "procedures": [
        {"procedure": "ECG", "cpt_code": None},
        {"procedure": "Troponin levels", "cpt_code": None}
    ],
    "vital_signs": {
        "blood_pressure": "145/92",
        "temperature": "98.6F",
        "heart_rate": "98",
        "respiratory_rate": "22",
        "oxygen_saturation": "96%"
    },
    "allergies": [],  # ❌ Missing
    "social_history": {
        "smoking": None,  # ❌ Missing
        "alcohol": None  # ❌ Missing
    }
}

sample_source_text = """PATIENT ENCOUNTER NOTE
Date: 02/10/2026
Patient: John Doe, 45M

CHIEF COMPLAINT: Chest pain

HISTORY OF PRESENT ILLNESS:
Patient presents to ED with chest pain. Reports pain started yesterday evening
while climbing stairs. Describes pain as pressure-like. Also reports some
shortness of breath and sweating. No nausea or vomiting. Patient appears anxious.

PAST MEDICAL HISTORY: Hypertension

MEDICATIONS: Takes blood pressure medication

VITAL SIGNS: BP: 145/92, HR: 98, Temp: 98.6F, RR: 22, SpO2: 96%

PHYSICAL EXAM:
Heart: Regular rhythm, no murmurs
Lungs: Clear bilaterally
Extremities: No edema

ASSESSMENT: Chest pain, likely cardiac in origin. Rule out acute coronary syndrome.

PLAN: ECG ordered. Troponin levels. Cardiology consult.
"""

print("✅ Sample document loaded")
print(f"   Document ID: {sample_doc_id}")
print(f"   Initial completeness: ~68%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Process Document (Agent Stages 1-2)

# COMMAND ----------

# Process document through ambiguity detection and question generation
result = agent.process_document(
    document_id=sample_doc_id,
    entities=sample_entities,
    source_text=sample_source_text
)

print("\n" + "="*60)
print("📋 PROCESSING RESULT")
print("="*60)
print(f"Status: {result['status']}")

if result['status'] == 'pending_review':
    print(f"\n✅ Questions Generated: {len(result['questions'])}")
    print(f"⏱️  Estimated Review Time: {result['estimated_review_time']} seconds\n")

    # Store questions for next step
    questions = result['questions']

    # Display questions
    print("QUESTIONS TO ANSWER:")
    print("-" * 60)
    for i, q in enumerate(questions, 1):
        print(f"\n📝 Q{i} (Priority {q['priority']} - {q.get('clinical_note', 'N/A')}):")
        print(f"   {q['text']}")
        print(f"   Context: {q.get('context', 'N/A')}")
        if q.get('options'):
            print(f"   Options: {', '.join(q['options'])}")
else:
    print(f"\n✅ Status: {result['status']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Create Widgets for Human Input
# MAGIC
# MAGIC **Databricks Widgets will appear at the top of this notebook.**
# MAGIC Answer the questions and run the next cell.

# COMMAND ----------

# Create widgets for each question
if result['status'] == 'pending_review':
    questions = result['questions']

    print("Creating input widgets...")
    print("📌 Widgets will appear at the TOP of this notebook")
    print("📌 Answer all questions, then run the next cell\n")

    # Remove existing widgets first
    try:
        for q in questions:
            dbutils.widgets.remove(q['id'])
    except:
        pass

    # Create new widgets
    for q in questions:
        q_id = q['id']
        question_text = f"Q{q['priority']}: {q['text']}"

        if q.get('input_type') == 'select' and q.get('options'):
            # Dropdown for select questions
            options = q['options']
            dbutils.widgets.dropdown(q_id, options[0], options, question_text)
            print(f"✅ Created dropdown for {q_id}")
        else:
            # Text input for open questions
            placeholder = q.get('placeholder', '')
            dbutils.widgets.text(q_id, '', question_text)
            print(f"✅ Created text input for {q_id}")

    print("\n" + "="*60)
    print("✅ WIDGETS CREATED!")
    print("="*60)
    print("👆 Look at the TOP of this notebook to answer questions")
    print("👉 After answering, run the NEXT cell to generate SOAP note")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Submit Feedback and Generate SOAP (Stages 3-4)

# COMMAND ----------

# Collect answers from widgets
if result['status'] == 'pending_review':
    answers = {}

    print("📥 Collecting answers from widgets...\n")

    for q in questions:
        q_id = q['id']
        answer = dbutils.widgets.get(q_id)

        if answer:  # Only include non-empty answers
            answers[q_id] = answer
            print(f"✅ {q_id}: {answer}")

    print(f"\n📊 Total answers collected: {len(answers)}/{len(questions)}")

    # Submit feedback and generate SOAP
    if answers:
        print("\n" + "="*60)
        print("🚀 GENERATING SOAP NOTE WITH YOUR FEEDBACK")
        print("="*60 + "\n")

        soap_result = agent.submit_feedback(
            document_id=sample_doc_id,
            answers=answers
        )

        # Display results
        print("\n" + "="*80)
        print("✅ SOAP NOTE GENERATED SUCCESSFULLY!")
        print("="*80 + "\n")

        print(soap_result['soap_note'])

        print("\n" + "="*80)
        print("📊 QUALITY METRICS")
        print("="*80)
        metrics = soap_result['quality_metrics']
        print(f"┌{'─'*38}┬{'─'*15}┬{'─'*15}┐")
        print(f"│ {'Metric':<36} │ {'Before':<13} │ {'After':<13} │")
        print(f"├{'─'*38}┼{'─'*15}┼{'─'*15}┤")
        print(f"│ {'Completeness':<36} │ {metrics['completeness_before']:>12.1%} │ {metrics['completeness_score']:>12.1%} │")
        print(f"│ {'Improvement':<36} │ {'─':<13} │ {metrics['completeness_improvement']:>13} │")
        print(f"│ {'Fields Improved':<36} │ {'─':<13} │ {metrics['fields_improved']:>13} │")
        print(f"│ {'Human Verified Fields':<36} │ {'─':<13} │ {metrics['human_verified_fields']:>13} │")
        print(f"│ {'Generation Time (seconds)':<36} │ {'─':<13} │ {metrics['generation_time_seconds']:>13.1f} │")
        print(f"│ {'Word Count':<36} │ {'─':<13} │ {metrics['word_count']:>13} │")
        print(f"└{'─'*38}┴{'─'*15}┴{'─'*15}┘")

        # Clean up widgets
        print("\n🧹 Cleaning up widgets...")
        for q in questions:
            try:
                dbutils.widgets.remove(q['id'])
            except:
                pass
        print("✅ Widgets removed\n")

        print("="*80)
        print("🎉 DEMO COMPLETE!")
        print("="*80)
        print("📊 View results in the cells below")

    else:
        print("\n⚠️  WARNING: No answers provided!")
        print("Please answer at least one question using the widgets at the top of the notebook.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 View Results in Gold Layer

# COMMAND ----------

# View Review Queue
print("📋 REVIEW QUEUE")
print("="*80)
display(spark.table(GOLD_REVIEW_QUEUE).orderBy(col("created_timestamp").desc()).limit(10))

# COMMAND ----------

# View Clinical Summaries (Final SOAP Notes)
print("📄 CLINICAL SUMMARIES (SOAP Notes)")
print("="*80)
display(spark.table(GOLD_CLINICAL_SUMMARIES).orderBy(col("generation_timestamp").desc()).limit(10))

# COMMAND ----------

# View Validated Entities
print("✅ VALIDATED ENTITIES")
print("="*80)
display(spark.table(GOLD_VALIDATED_ENTITIES).orderBy(col("human_review_timestamp").desc()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 MLflow Tracking

# COMMAND ----------

# View MLflow runs
import mlflow

print("🔬 MLFLOW EXPERIMENT TRACKING")
print("="*80)

# Get current experiment
experiment = mlflow.get_experiment_by_name("/Shared/interactive-soap-agent")
if not experiment:
    experiment = mlflow.get_experiment_by_name(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/interactive-soap-agent")

if experiment:
    print(f"Experiment: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}\n")

    # Get recent runs
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10
    )

    if not runs_df.empty:
        print("📊 Recent Runs:")
        display(runs_df[['run_id', 'start_time', 'tags.mlflow.runName',
                        'metrics.processing_time_seconds', 'metrics.improvement_delta']])
    else:
        print("No runs found yet. Runs will appear after processing documents.")
else:
    print("Experiment will be created automatically when processing documents.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ✅ SETUP COMPLETE!
# MAGIC
# MAGIC ### What You Just Built:
# MAGIC
# MAGIC 1. ✅ **Ambiguity Detection Agent** using Llama 3.3 70B
# MAGIC 2. ✅ **Question Generation Agent** using GPT-5-2
# MAGIC 3. ✅ **Feedback Processing** (pure Python)
# MAGIC 4. ✅ **SOAP Generation Agent** using GPT-5-2
# MAGIC 5. ✅ **Gold Layer Tables** (review queue, clinical summaries, validated entities)
# MAGIC 6. ✅ **MLflow Tracking** for all agent operations
# MAGIC 7. ✅ **Interactive Widgets** for human-in-the-loop review
# MAGIC
# MAGIC ### Model Performance:
# MAGIC
# MAGIC | Agent Stage | Model | Strength |
# MAGIC |-------------|-------|----------|
# MAGIC | Ambiguity Detection | Llama 3.3 70B | Excellent reasoning, cost-effective |
# MAGIC | Question Generation | GPT-5-2 | Best natural language, context understanding |
# MAGIC | SOAP Generation | GPT-5-2 | Professional medical writing |
# MAGIC
# MAGIC ### Next Steps:
# MAGIC
# MAGIC 1. **Process Your Own PDFs**: Replace sample data with your clinical documents
# MAGIC 2. **Tune Prompts**: Customize prompts in agent classes for your use case
# MAGIC 3. **Batch Processing**: Process multiple documents from your Silver layer
# MAGIC 4. **Production Deployment**: Set up scheduled workflows
# MAGIC 5. **Quality Analysis**: Track improvement metrics over time
# MAGIC
# MAGIC ### Cost Optimization:
# MAGIC
# MAGIC To reduce costs, you can switch to smaller models:
# MAGIC ```python
# MAGIC AMBIGUITY_MODEL = "databricks-meta-llama-3-1-8b-instruct"  # Smaller, faster
# MAGIC QUESTION_MODEL = "databricks-gpt-5-1"                       # Previous version
# MAGIC SOAP_MODEL = "databricks-gpt-5-1"                           # Previous version
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎉 Congratulations! Your Interactive SOAP Agent is ready for your hackathon!**
