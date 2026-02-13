from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import json
import mlflow

class ClinicalEntityExtractor:
    """
    Extract clinical entities using Databricks DBRX Instruct
    """

    def __init__(self, model_endpoint="databricks-dbrx-instruct"):
        self.w = WorkspaceClient()
        self.model_endpoint = model_endpoint

        # Load prompts from MLflow
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        """Load versioned prompts from MLflow"""
        return {
            'extraction': """You are a medical AI assistant specialized in clinical documentation.

Extract structured clinical information from this note. Be precise and only extract information explicitly stated.

CLINICAL NOTE:
{clinical_text}

Extract the following entities in JSON format:

{{
  "chief_complaint": "primary reason for visit (single sentence)",
  "symptoms": [
    {{
      "symptom": "symptom name",
      "onset": "when started",
      "severity": "mild/moderate/severe",
      "location": "body location if applicable"
    }}
  ],
  "diagnoses": [
    {{
      "diagnosis": "diagnosis name",
      "icd10_code": "ICD-10 code if mentioned",
      "status": "confirmed/suspected/differential"
    }}
  ],
  "medications": [
    {{
      "name": "medication name",
      "dosage": "dosage if specified",
      "frequency": "frequency if specified",
      "route": "route of administration"
    }}
  ],
  "procedures": [
    {{
      "procedure": "procedure name",
      "cpt_code": "CPT code if mentioned"
    }}
  ],
  "vital_signs": {{
    "blood_pressure": "value",
    "temperature": "value with unit",
    "heart_rate": "value",
    "respiratory_rate": "value",
    "oxygen_saturation": "value"
  }},
  "allergies": ["allergy1", "allergy2"],
  "social_history": {{
    "smoking": "status",
    "alcohol": "status",
    "occupation": "if mentioned"
  }},
  "extraction_confidence": "high/medium/low",
  "missing_information": ["list of critical missing data"]
}}

IMPORTANT:
- Only extract information explicitly stated in the text
- Do not infer or assume information
- If a field is not mentioned, use null
- Use proper medical terminology
- Return ONLY valid JSON, no other text

JSON OUTPUT:"""
        }

    def extract_entities(self, clinical_text: str, sections: dict = None) -> dict:
        """
        Extract clinical entities from text using DBRX

        Args:
            clinical_text: Full clinical note text
            sections: Optional structured sections from parser

        Returns:
            Extracted entities as structured dict
        """

        # Log prompt to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_endpoint", self.model_endpoint)
            mlflow.log_param("text_length", len(clinical_text))

            # Prepare prompt
            prompt = self.prompts['extraction'].format(clinical_text=clinical_text)

            try:
                # Call DBRX via Foundation Model API
                response = self.w.serving_endpoints.query(
                    name=self.model_endpoint,
                    messages=[ChatMessage(
                        role=ChatMessageRole.USER,
                        content=prompt
                    )],
                    max_tokens=2000,
                    temperature=0.1,  # Low temp for factual extraction
                    top_p=0.95
                )

                # Parse response
                extracted_text = response.choices[0].message.content

                # Clean and parse JSON
                entities = self._parse_json_response(extracted_text)

                # Log results
                mlflow.log_dict(entities, "extracted_entities.json")
                mlflow.log_metric("num_symptoms", len(entities.get('symptoms', [])))
                mlflow.log_metric("num_diagnoses", len(entities.get('diagnoses', [])))
                mlflow.log_metric("num_medications", len(entities.get('medications', [])))

                return entities

            except Exception as e:
                mlflow.log_param("error", str(e))
                raise

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from LLM response, handling common issues"""

        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # Attempt to fix common JSON issues
            # (add more sophisticated error handling as needed)
            raise ValueError(f"Failed to parse JSON: {str(e)}\n{response_text}")

    def extract_batch(self, documents: list) -> list:
        """Extract entities from multiple documents in parallel"""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.extract_entities, doc['text']) for doc in documents]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        return results