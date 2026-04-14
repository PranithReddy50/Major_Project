import markdown
from xhtml2pdf import pisa
import os

# Paths
ARTIFACT_PATH = r"C:\Users\vinay\.gemini\antigravity\brain\ed21d168-59c8-42bb-ab5d-e53db2af1fa1\project_submission.md"
OUTPUT_PATH = "Project_Submission_Report.pdf"

def convert_md_to_pdf(md_path, pdf_path):
    # 1. Read Markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Convert to HTML
    html = markdown.markdown(text, extensions=['fenced_code', 'codehilite', 'tables'])

    # 3. Add Basic CSS for readability
    css = """
    <style>
        body { font-family: Helvetica, sans-serif; font-size: 10pt; line-height: 1.5; color: #333; }
        h1 { color: #154360; font-size: 24pt; border-bottom: 2px solid #154360; padding-bottom: 10px; margin-top: 20px; }
        h2 { color: #2E86C1; font-size: 18pt; margin-top: 15px; border-bottom: 1px solid #ddd; }
        h3 { color: #1F618D; font-size: 14pt; margin-top: 10px; }
        h4 { color: #5D6D7E; font-size: 12pt; font-weight: bold; }
        pre, code { font-family: "Courier New", Courier, monospace; background-color: #f4f4f4; padding: 2px; border: 1px solid #ddd; border-radius: 3px; font-size: 9pt; }
        pre { white-space: pre-wrap; word-wrap: break-word; padding: 10px; margin: 10px 0; background-color: #f8f8f8; }
        p { margin-bottom: 10px; }
        li { margin-bottom: 5px; }
        hr { border: 0; height: 1px; background: #ccc; margin: 20px 0; }
        @page { size: A4; margin: 1.5cm; }
    </style>
    """
    
    full_html = f"<html><head>{css}</head><body>{html}</body></html>"

    # 4. Generate PDF
    with open(pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        print("Error converting to PDF")
    else:
        print(f"Successfully created: {os.path.abspath(pdf_path)}")

if __name__ == "__main__":
    if not os.path.exists(ARTIFACT_PATH):
        print(f"Artifact not found at {ARTIFACT_PATH}")
    else:
        convert_md_to_pdf(ARTIFACT_PATH, OUTPUT_PATH)
