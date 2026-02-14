from fpdf import FPDF
import datetime
import os

def generate_sar_pdf(account_id, risk_score, investigation_summary, output_dir="sar_reports"):
    """
    Generates a formal Suspicious Activity Report (SAR) in PDF format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, "Suspicious Activity Report (SAR)", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(190, 10, f"Generated Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    
    # Case Details
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "1. CASE IDENTIFICATION", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(190, 10, f"Account Holder ID: {account_id}", ln=True)
    pdf.cell(190, 10, f"Initial ML Risk Score: {risk_score}", ln=True)
    pdf.ln(5)
    
    # Investigation Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "2. INVESTIGATION NARRATIVE", ln=True)
    pdf.set_font("Arial", '', 10)
    
    # Split summary by lines to handle PDF wrapping
    pdf.multi_cell(190, 7, investigation_summary)
    pdf.ln(10)
    
    # Compliance Verdict
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "3. REGULATORY COMPLIANCE VERDICT", ln=True)
    pdf.set_font("Arial", 'I', 10)
    verdict = "HIGH" if risk_score > 0.9 else "MEDIUM/LOW"
    pdf.multi_cell(190, 7, f"Based on the agentic investigation, this case has been classified as {verdict} priority for human review.")
    
    # Footer
    filename = f"SAR_Account_{account_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(output_dir, filename)
    pdf.output(filepath)
    
    return filepath
