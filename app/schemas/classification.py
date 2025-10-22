from pydantic import BaseModel, Field


class ClassificationOutput(BaseModel):
    """Schema for structured extraction of insurance form details."""

    quote_submission: bool = Field(
        description="Returns true if the PDF file is a valid insurance quote form or application. Else returns false."
    )
    quote_class: str = Field(
        description="Returns 'personal' for personal auto and property quote requests or 'commercial' for commercial auto and property quote request forms. Return '' if the class is not clearly determined."
    )
    quote_type: str = Field(
        description="Return 'auto' for all types of auto quotes (both personal and commercial) and return 'property' for all sorts of property quotes (including Homeowners, Commercial Property, etc.). Return 'both' if it is an umbrella quote request containing both auto and property information. Return '' if nothing is found."
    )
    acord_form: bool = Field(
        description="Return true if the form is one of the ACORD standard forms (e.g., ACORD 125, 88). Else return false."
    )
    acord_form_id: str = Field(
        description="If the form is a standard ACORD form, return its standard form number (e.g., 'ACORD 125', 'ACORD 80') exactly as found on the document. Else return ''."
    )
