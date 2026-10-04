"""Generates the small, original demo notes used in examples/ (no third-party content)."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont
import zipfile

OUT = Path(__file__).resolve().parents[1] / "examples" / "sample_notes"
OUT.mkdir(parents=True, exist_ok=True)

PDF_LINES = """SQL & Spark Notes

1. What is the difference between WHERE and HAVING?
Answer: WHERE filters rows before grouping; HAVING filters groups after GROUP BY, so it can use aggregates.

2. Find the second highest salary from the employees table.
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

3. Find the second lowest salary from the employees table.
SELECT MIN(salary) AS second_lowest
FROM employees
WHERE salary > (SELECT MIN(salary) FROM employees);

4. What is the difference between ROW_NUMBER, RANK and DENSE_RANK?
Answer: ROW_NUMBER gives 1,2,3 even for ties; RANK gives ties the same number and leaves gaps (1,1,3);
DENSE_RANK gives ties the same number without gaps (1,1,2).

5. What is a broadcast join in Spark?
Answer: The small table is copied to every executor so the large table does not need to be shuffled.
Use it when one side comfortably fits in memory.

6. Why is data skew a problem in Spark?
Answer: A few keys hold most rows, so one task runs far longer than the rest. Fix with AQE skew join,
salting, or broadcasting the smaller side.

7. Explain the Medallion architecture.
Answer: Bronze keeps raw data as received, Silver holds cleaned and de-duplicated data, and Gold holds
business-ready aggregates for BI and ML.""".split("\n")

def make_pdf():
    c = canvas.Canvas(str(OUT / "sql_spark_notes.pdf"), pagesize=A4)
    y = 800
    for line in PDF_LINES:
        c.setFont("Helvetica-Bold" if line[:2].strip().rstrip(".").isdigit() else "Helvetica", 10)
        c.drawString(50, y, line); y -= 15
        if y < 60: c.showPage(); y = 800
    c.save()

TXT = """ADF open questions (answers still to write)

How would you design a metadata-driven pipeline that loads 100 tables without creating 100 pipelines?
What is the difference between a linked service and a dataset?
How would you implement an incremental load using a watermark column?
A daily file does not arrive on time. How would you alert instead of failing the pipeline?
What is the difference between ROW_NUMBER, RANK and DENSE_RANK?
Tell me about yourself.
"""

IMG_TEXT = """Delta Lake cheat sheet

Q1. What is Delta Lake?
A: Parquet data files plus a transaction log (_delta_log)
that adds ACID transactions, schema enforcement and time travel.

Q2. What does VACUUM do?
A: Deletes data files no longer referenced by the log,
older than the retention period (default 7 days).

Q3. What is the Medallion architecture?
A: Bronze = raw, Silver = cleaned, Gold = business-ready."""

def make_img():
    img = Image.new("RGB", (1100, 700), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    y = 30
    for line in IMG_TEXT.split("\n"):
        d.text((40, y), line, fill="black", font=f); y += 38
    img.save(OUT / "delta_cheatsheet.png")

if __name__ == "__main__":
    make_pdf(); (OUT / "adf_open_questions.txt").write_text(TXT); make_img()
    with zipfile.ZipFile(OUT.parent / "sample_notes.zip", "w") as z:
        for f in sorted(OUT.iterdir()): z.write(f, f.name)
    print("sample notes written to", OUT)
