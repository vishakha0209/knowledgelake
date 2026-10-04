"""Generates a second, non-technical demo set: a fictional company's IT helpdesk and HR documents.

It shows that knowledgelake is not tied to data engineering: run it with the default
`--topics auto` and it discovers the topics (accounts, VPN, leave, expenses...) by itself.
All content is original and the company ("Northwind Labs") is fictional.

Same mess as real life: a text PDF, a Q:/A: text file, a screenshot that needs OCR, a
question-only list, and the same questions reworded across files.
"""
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "examples" / "workplace_docs"
OUT.mkdir(parents=True, exist_ok=True)

FAQ_PDF = """Northwind Labs - IT Helpdesk FAQ

1. How do I reset my password?
Answer: Go to the self-service portal, choose Forgot password, and verify with your authenticator app.
Passwords must be at least 14 characters and are valid for 180 days.

2. My account is locked. What should I do?
Answer: Accounts lock after 5 failed sign-ins and unlock automatically after 15 minutes.
If you need access sooner, call the helpdesk and verify your identity.

3. How do I set up multi-factor authentication (MFA)?
Answer: Install the authenticator app, open the security page in the portal and scan the QR code.
Keep the backup codes somewhere safe, not on your laptop.

4. How do I connect to the VPN?
Answer: Open the VPN client, choose the nearest region and sign in with your work account and MFA.
The VPN is required for internal tools such as finance and HR systems.

5. The VPN keeps disconnecting. How can I fix it?
Answer: Switch to a wired connection or a 5 GHz network, update the VPN client, and pick a closer region.
If it still drops every few minutes, send the client log to the helpdesk.

6. How do I request new software?
Answer: Raise a software request in the service catalogue. Licensed tools need manager approval;
free tools on the approved list install automatically within one working day.

7. My laptop is very slow. What can I try?
Answer: Restart it (many issues come from weeks without a reboot), close browser tabs, check that disk
space is above 15 percent, and install pending updates. If that fails, book a hardware check.

8. How do I report a phishing email?
Answer: Use the Report phishing button in the mail client. Do not click links or open attachments.
If you already clicked, tell the security team immediately; it is never too late to report.

9. How do I get a replacement laptop charger?
Answer: Collect one from the IT desk on any floor, or raise a hardware request for delivery.

10. How do I set up email on my phone?
Answer: Install the official mail app, sign in with your work account and accept the device policy.
Personal phones are supported if the screen lock is enabled.

11. Can I use a personal USB drive?
Answer: No. USB storage is blocked on company laptops. Share files through the company drive instead.

12. How long does the helpdesk take to respond?
Answer: Urgent issues (nobody can work) within 1 hour; normal requests within 1 working day.""".split("\n")

HR_TXT = """Northwind Labs - People & Policies

Q: How many days of annual leave do I get?
A: Full-time employees get 24 days of annual leave per year, plus public holidays. Unused leave
up to 5 days carries over to the next year.

Q: How do I apply for leave?
A: Submit the request in the HR portal at least two weeks ahead for leave longer than three days.
Your manager approves it in the portal.

Q: What is the sick leave policy?
A: Up to 10 paid sick days per year. For more than 3 days in a row, upload a doctor's note.

Q: When is salary paid?
A: Salary is paid on the last working day of every month. Payslips appear in the HR portal two days before.

Q: How do I update my bank details for payroll?
A: Change them in the HR portal before the 20th of the month so the new account is used for that month's pay.

Q: What is the parental leave policy?
A: Primary caregivers get 26 weeks of paid leave and secondary caregivers get 6 weeks, available to
all employees after 6 months of service.

Q: Can I work from home?
A: Most roles can work from home up to 3 days a week, agreed with your manager. Some roles need to be on site.

Q: How do I claim travel expenses?
A: Submit the claim in the expense tool within 30 days with a photo of each receipt. Claims over 500
need a pre-approved travel request.

Q: What is the daily meal allowance when travelling?
A: Up to 40 per day for domestic travel and up to 60 per day abroad. Alcohol is not reimbursed.

Q: Can I book business class?
A: Only for flights longer than 8 hours, and only with director approval.

Q: How do I claim my home internet costs?
A: Employees who work from home at least 2 days a week can claim up to 30 per month through the expense tool.

Q: Where do I find the learning budget?
A: Every employee has a learning budget of 1,000 per year for courses, books and conferences.
Request it in the HR portal with a short note on how it helps your role."""

IMG_TEXT = """IT quick card - print me!

Q1. Forgot your password?
A: Self-service portal -> Forgot password -> verify with MFA.

Q2. VPN dropping all the time?
A: Use wired or 5 GHz Wi-Fi, update the client, pick a closer region.

Q3. Got a suspicious email?
A: Press Report phishing. Never click links in it.

Q4. Laptop slow?
A: Restart first, then free disk space and install updates."""

QUESTIONS_TXT = """Questions from new joiners (answers to add)

How do I get access to the shared team drive?
Who do I contact if my badge stops working?
How do I book a meeting room?
How many days of annual leave do I get?
Where can I see my payslip?
How do I reset my password?
"""


def make_pdf():
    c = canvas.Canvas(str(OUT / "it_helpdesk_faq.pdf"), pagesize=A4)
    y = 800
    for line in FAQ_PDF:
        c.setFont("Helvetica-Bold" if line[:2].strip().rstrip(".").isdigit() else "Helvetica", 10)
        c.drawString(50, y, line)
        y -= 15
        if y < 60:
            c.showPage()
            y = 800
    c.save()


def make_img():
    img = Image.new("RGB", (1100, 640), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    y = 30
    for line in IMG_TEXT.split("\n"):
        d.text((40, y), line, fill="black", font=f)
        y += 38
    img.save(OUT / "it_quick_card.png")


if __name__ == "__main__":
    make_pdf()
    (OUT / "hr_policies.txt").write_text(HR_TXT)
    (OUT / "new_joiner_questions.txt").write_text(QUESTIONS_TXT)
    make_img()
    with zipfile.ZipFile(OUT.parent / "workplace_docs.zip", "w") as z:
        for f in sorted(OUT.iterdir()):
            z.write(f, f.name)
    print("workplace demo documents written to", OUT)
