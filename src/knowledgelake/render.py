"""Stage 7 - Render one styled PDF per topic (cover, clickable contents, sections, code boxes).

Answers support ```code fences``` and simple bullet lines. LLM-written answers get a ✎ marker."""
import json, glob, os, re, sys, datetime
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Preformatted,
                                Table, TableStyle, PageBreak, KeepTogether, Flowable)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

F = '/usr/share/fonts/truetype/dejavu/'
pdfmetrics.registerFont(TTFont('Sans', F + 'DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('Sans-Bold', F + 'DejaVuSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Sans-Oblique', F + 'DejaVuSans-Oblique.ttf'))
pdfmetrics.registerFont(TTFont('Mono', F + 'DejaVuSansMono.ttf'))
from reportlab.lib.fonts import addMapping
addMapping('Sans', 0, 0, 'Sans'); addMapping('Sans', 1, 0, 'Sans-Bold'); addMapping('Sans', 0, 1, 'Sans-Oblique'); addMapping('Sans', 1, 1, 'Sans-Bold')

INK = colors.HexColor('#1F2937'); ACC = colors.HexColor('#1D4ED8'); MUTED = colors.HexColor('#6B7280')
CODEBG = colors.HexColor('#F3F4F6'); LINE = colors.HexColor('#E5E7EB'); TAG = colors.HexColor('#B45309')

ST = {
 'title': ParagraphStyle('t', fontName='Sans-Bold', fontSize=28, leading=34, textColor=INK),
 'subtitle': ParagraphStyle('st', fontName='Sans', fontSize=12, leading=17, textColor=MUTED),
 'h1': ParagraphStyle('h1', fontName='Sans-Bold', fontSize=16, leading=21, textColor=ACC, spaceBefore=6, spaceAfter=8),
 'q': ParagraphStyle('q', fontName='Sans-Bold', fontSize=10.5, leading=14.5, textColor=INK, spaceBefore=2, spaceAfter=3),
 'a': ParagraphStyle('a', fontName='Sans', fontSize=9.3, leading=13.3, textColor=INK, spaceAfter=3),
 'bul': ParagraphStyle('b', fontName='Sans', fontSize=9.3, leading=13.3, textColor=INK, leftIndent=10, bulletIndent=0, spaceAfter=1),
 'code': ParagraphStyle('c', fontName='Mono', fontSize=7.8, leading=10.2, textColor=INK),
 'meta': ParagraphStyle('m', fontName='Sans', fontSize=7.3, leading=10, textColor=MUTED, spaceAfter=2),
 'small': ParagraphStyle('s', fontName='Sans', fontSize=9, leading=13, textColor=INK),
 'toc1': ParagraphStyle('toc1', fontName='Sans', fontSize=10, leading=15, leftIndent=0, textColor=INK),
}

class Rule(Flowable):
    def __init__(self, w=None): super().__init__(); self.w = w
    def wrap(self, aw, ah): self.aw = aw; return aw, 6
    def draw(self): self.canv.setStrokeColor(LINE); self.canv.setLineWidth(0.6); self.canv.line(0, 3, self.aw, 3)

def inline(s):
    s = escape(s)
    s = re.sub(r'`([^`\n]+)`', r'<font name="Mono" size="8.3">\1</font>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
    return s

def code_block(txt):
    lines = []
    for l in txt.rstrip('\n').split('\n'):
        while len(l) > 98:  # hard wrap long lines
            lines.append(l[:98]); l = '    ' + l[98:]
        lines.append(l)
    pre = Preformatted('\n'.join(lines), ST['code'])
    t = Table([[pre]], colWidths=['100%'])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), CODEBG), ('BOX', (0, 0), (-1, -1), 0.4, LINE),
                           ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                           ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    return [t, Spacer(1, 3)]

def answer_flowables(a):
    out = []
    parts = re.split(r'```[a-zA-Z]*\n?', a)
    for k, part in enumerate(parts):
        if k % 2 == 1:
            if part.strip(): out += code_block(part)
            continue
        for para in re.split(r'\n\s*\n', part):
            para = para.strip('\n')
            if not para.strip(): continue
            buf = []
            for line in para.split('\n'):
                ls = line.strip()
                if not ls: continue
                m = re.match(r'^([•\-\*▪◦✓✔]|\d+[.)])\s+(.*)', ls)
                if m:
                    if buf: out.append(Paragraph(inline(' '.join(buf)), ST['a'])); buf = []
                    b = '•' if not m.group(1)[0].isdigit() else m.group(1)
                    out.append(Paragraph(inline(m.group(2)), ST['bul'], bulletText=b))
                else:
                    buf.append(ls)
            if buf: out.append(Paragraph(inline('\n'.join(buf)).replace('\n', '<br/>'), ST['a']))
    return out

class Doc(BaseDocTemplate):
    def __init__(self, fn, topic, **kw):
        super().__init__(fn, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=16*mm,
                         title=f'{topic} – Knowledge Base', author='knowledgelake', **kw)
        self.topic = topic
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id='f')
        self.addPageTemplates([PageTemplate('p', [fr], onPage=self.deco)])
    def deco(self, c, d):
        if d.page == 1: return
        c.saveState(); c.setFont('Sans', 7.5); c.setFillColor(MUTED)
        c.drawString(18*mm, 10*mm, f'{self.topic} · knowledgelake')
        c.drawRightString(A4[0]-18*mm, 10*mm, str(d.page)); c.restoreState()
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == 'h1':
            txt = f.getPlainText(); key = 'k%d' % id(f)
            self.canv.bookmarkPage(key); self.canv.addOutlineEntry(txt, key, 0)
            self.notify('TOCEntry', (0, txt, self.page, key))

def llm(it): return it.get('answered_by_llm') or it.get('answered_by_claude')


def build(path, out_dir, title_suffix='Knowledge Base'):
    d = json.load(open(path)); topic = d['topic']; items = [i for i in d['items']]
    order = d.get('section_order') or []
    secs = order + sorted({i['section'] for i in items} - set(order))
    fn = os.path.join(out_dir, re.sub(r'[^A-Za-z0-9]+', '_', topic).strip('_') + '_Knowledge_Base.pdf')
    doc = Doc(fn, topic)
    srcs = {}
    for it in items:
        for s in it['sources']: srcs[s] = srcs.get(s, 0) + 1
    n_cl = sum(1 for i in items if llm(i))
    last = max((i.get('added') or '') for i in items) or '-'
    story = [Spacer(1, 40*mm), Paragraph(escape(topic), ST['title']), Spacer(1, 4*mm),
             Paragraph('Questions and answers from many sources — de-duplicated and organised by section', ST['subtitle']), Spacer(1, 10*mm)]
    rows = [['Questions', str(len(items))], ['Sections', str(len([s for s in secs if any(i['section']==s for i in items)]))],
            ['AI-written answers', f'{n_cl} (marked ✎ — the source had the question only)'], ['Last updated', last]]
    t = Table(rows, colWidths=[58*mm, 105*mm])
    t.setStyle(TableStyle([('FONT', (0, 0), (0, -1), 'Sans-Bold', 9.5), ('FONT', (1, 0), (1, -1), 'Sans', 9.5),
                           ('TEXTCOLOR', (0, 0), (-1, -1), INK), ('LINEBELOW', (0, 0), (-1, -1), 0.4, LINE),
                           ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    story += [t, Spacer(1, 8*mm), Paragraph('<b>Sources merged into this file</b>', ST['small']), Spacer(1, 2*mm)]
    for s, n in sorted(srcs.items(), key=lambda x: -x[1]):
        story.append(Paragraph(f'• {escape(s)} — {n}', ST['meta']))
    story += [PageBreak(), Paragraph('Contents', ST['h1'].clone('ct', textColor=INK))]
    toc = TableOfContents(); toc.levelStyles = [ST['toc1']]; toc.dotsMinLevel = 0
    story += [toc, PageBreak()]
    qn = 0
    for s in secs:
        its = [i for i in items if i['section'] == s]
        if not its: continue
        story.append(Paragraph(escape(s) + f'  <font size="10" color="#6B7280">({len(its)})</font>', ST['h1']))
        for it in its:
            qn += 1
            mark = ' <font color="#B45309">✎</font>' if llm(it) else ''
            lvl = f'  <font size="7.5" color="#6B7280">[{it["level"]}]</font>' if it.get('level') else ''
            head = [Paragraph(f'<font color="#1D4ED8">Q{qn}.</font> ' + inline(it['q']) + lvl + mark, ST['q'])]
            body = answer_flowables(it['a']) or [Paragraph('<i>No answer yet — run the answer step or add one.</i>', ST['meta'])]
            src = '; '.join(it['sources'])
            if llm(it): src = 'AI-written answer · question from ' + src
            meta = [Paragraph('Source: ' + escape(src), ST['meta']), Rule(), Spacer(1, 2)]
            if body: story.append(KeepTogether(head + body[:1]))
            else: story += head
            story += body[1:] + meta
    doc.multiBuild(story)
    return fn, len(items)

def build_all(kb_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for p in sorted(glob.glob(os.path.join(str(kb_dir), '*.json'))):
        with open(p) as f:
            d = json.load(f)
        if isinstance(d, dict) and 'topic' in d and 'items' in d:   # skip kb_config.json and other files
            out.append(build(p, out_dir))
    return out
