from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def generate_pdf(applicant_data, output_folder="outputs/reports"):
    os.makedirs(output_folder, exist_ok=True)

    name = f"applicant_{applicant_data.name}"
    filename = os.path.join(output_folder, f"{name}.pdf")

    c = canvas.Canvas(filename, pagesize=A4)
    textobject = c.beginText(40, 800)

    textobject.textLine("Smart Credit Risk Engine – Loan Decision Report")
    textobject.textLine(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    textobject.textLine("--------------------------------------------------")

    for key, value in applicant_data.to_dict().items():
        textobject.textLine(f"{key}: {value}")

    c.drawText(textobject)
    c.save()

    return filename
