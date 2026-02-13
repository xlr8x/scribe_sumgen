"""
Clinical PDF Parser Module

High-accuracy PDF parsing optimized for clinical documents using unstructured.io
"""

from typing import Dict, List, Optional
import logging
import io

# Import dependencies only when needed to avoid import errors
# This allows fallback to PyMuPDF if unstructured is not installed


class ClinicalPDFParser:
    """
    High-accuracy PDF parser optimized for clinical documents.
    Uses unstructured.io with layout preservation for medical forms.
    """

    def __init__(self, strategy: str = "hi_res"):
        """
        Initialize PDF parser.

        Args:
            strategy: Parsing strategy - "hi_res" for high accuracy, "fast" for speed
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

    def parse_pdf(self, pdf_path: str) -> Dict:
        """
        Parse PDF and extract structured content.

        Args:
            pdf_path: Path to PDF file (local or DBFS)

        Returns:
            Dict with:
                - text: Full extracted text
                - sections: List of document sections
                - tables: List of extracted tables
                - metadata: Document metadata
        """
        try:
            # Import here to avoid issues if not installed
            from unstructured.partition.pdf import partition_pdf

            # Parse PDF with layout awareness
            elements = partition_pdf(
                filename=pdf_path,
                strategy=self.strategy,
                infer_table_structure=True,
                include_page_breaks=True,
                extract_images_in_pdf=False  # Focus on text for clinical notes
            )

            # Structure the output
            parsed_data = {
                'text': '\n'.join([str(el) for el in elements]),
                'sections': self._extract_sections(elements),
                'tables': self._extract_tables(elements),
                'metadata': self._extract_metadata(elements)
            }

            return parsed_data

        except ImportError:
            self.logger.warning("unstructured library not available, using fallback")
            return self._fallback_parse(pdf_path)
        except Exception as e:
            self.logger.error(f"PDF parsing failed: {str(e)}")
            raise

    def parse_pdf_bytes(self, pdf_bytes: bytes) -> Dict:
        """
        Parse PDF from bytes (useful for Spark UDFs).

        Args:
            pdf_bytes: PDF file as bytes

        Returns:
            Dict with parsed content
        """
        import tempfile
        import os

        # Write bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            # Parse the temporary file
            result = self.parse_pdf(tmp_path)
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_sections(self, elements) -> List[Dict]:
        """
        Extract document sections (e.g., Chief Complaint, History).

        Args:
            elements: Parsed elements from unstructured

        Returns:
            List of section dictionaries
        """
        sections = []
        current_section = None

        for el in elements:
            # Identify section headers
            if el.category == "Title":
                # Save previous section
                if current_section:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    'title': str(el),
                    'content': []
                }

            elif current_section and el.category in ["NarrativeText", "ListItem"]:
                # Add content to current section
                current_section['content'].append(str(el))

        # Add final section
        if current_section:
            sections.append(current_section)

        return sections

    def _extract_tables(self, elements) -> List[Dict]:
        """
        Extract tables (vital signs, lab results, etc.).

        Args:
            elements: Parsed elements from unstructured

        Returns:
            List of table dictionaries
        """
        tables = []

        for el in elements:
            if el.category == "Table":
                table_data = {
                    'data': el.metadata.text_as_html if hasattr(el.metadata, 'text_as_html') else str(el),
                    'type': 'clinical_data',
                    'text': str(el)
                }
                tables.append(table_data)

        return tables

    def _extract_metadata(self, elements) -> Dict:
        """
        Extract document metadata.

        Args:
            elements: Parsed elements from unstructured

        Returns:
            Metadata dictionary
        """
        # Count pages
        page_numbers = set()
        for el in elements:
            if hasattr(el.metadata, 'page_number') and el.metadata.page_number:
                page_numbers.add(el.metadata.page_number)

        # Count sections and tables
        num_sections = len([el for el in elements if el.category == "Title"])
        num_tables = len([el for el in elements if el.category == "Table"])

        return {
            'num_pages': len(page_numbers) if page_numbers else 0,
            'num_sections': num_sections,
            'num_tables': num_tables,
            'total_elements': len(elements)
        }

    def _fallback_parse(self, pdf_path: str) -> Dict:
        """
        Fallback PDF parsing using PyMuPDF if unstructured is not available.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with parsed content
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            text_content = []

            for page in doc:
                text_content.append(page.get_text())

            full_text = '\n'.join(text_content)

            return {
                'text': full_text,
                'sections': [],
                'tables': [],
                'metadata': {
                    'num_pages': len(doc),
                    'num_sections': 0,
                    'num_tables': 0,
                    'parser': 'fallback_pymupdf'
                }
            }

        except ImportError:
            self.logger.error("Neither unstructured nor PyMuPDF available")
            raise ImportError("No PDF parsing library available. Install 'unstructured' or 'PyMuPDF'")
        except Exception as e:
            self.logger.error(f"Fallback parsing failed: {str(e)}")
            raise


def calculate_parse_quality(parsed_data: Dict) -> float:
    """
    Calculate quality score for parsed document.

    Args:
        parsed_data: Parsed document data

    Returns:
        Quality score (0.0 to 1.0)
    """
    score = 0.0

    # Has text content
    if parsed_data.get('text') and len(parsed_data['text']) > 100:
        score += 0.4

    # Has sections
    if len(parsed_data.get('sections', [])) > 0:
        score += 0.3

    # Has tables
    if len(parsed_data.get('tables', [])) > 0:
        score += 0.2

    # Has metadata
    metadata = parsed_data.get('metadata', {})
    if metadata.get('num_pages', 0) > 0:
        score += 0.1

    return min(score, 1.0)


# Example usage
if __name__ == "__main__":
    parser = ClinicalPDFParser()

    # Parse a PDF
    # result = parser.parse_pdf("sample_clinical_note.pdf")
    # print(f"Extracted {len(result['text'])} characters")
    # print(f"Found {len(result['sections'])} sections")
    # print(f"Quality score: {calculate_parse_quality(result)}")
