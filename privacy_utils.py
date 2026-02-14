import re

def deidentify_data(text: str):
    """
    Replaces sensitive account numbers with placeholders to protect PII.
    Example: 'Account 12345678' -> 'Account [REDACTED_ACCOUNT]'
    """
    # Simple regex for 8-12 digit account numbers
    deidentified = re.sub(r'\b\d{8,12}\b', '[REDACTED_ACCOUNT]', text)
    return deidentified

def reidentify_report(report: str, mapping: dict):
    """Optional: Swaps placeholders back for the final internal SAR."""
    for placeholder, real_value in mapping.items():
        report = report.replace(placeholder, str(real_value))
    return report
