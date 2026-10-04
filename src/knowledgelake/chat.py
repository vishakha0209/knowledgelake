"""`kb chat` - a small local web app: ask a question, get an answer grounded in your notes.

    kb chat --kb kb_store            # then open http://127.0.0.1:8501

GET  /             the page
GET  /api/topics   topic list for the filter
POST /api/ask      {"question": "...", "topic": "" }  ->  {"answer", "generated", "hits": [...]}
"""
from __future__ import annotations

import logging
from pathlib import Path

from .rag import Retriever, answer
from .store import KnowledgeBase

log = logging.getLogger(__name__)


def create_app(kb_dir: Path, retriever: Retriever | None = None):
    from flask import Flask, jsonify, request

    kb = KnowledgeBase(Path(kb_dir))
    items = kb.items
    if not items:
        raise SystemExit(f"knowledge base {kb_dir} is empty - run `kb ingest` first")
    r = retriever or Retriever(items)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.get("/api/topics")
    def topics():
        return jsonify(sorted({i.topic for i in items}))

    @app.post("/api/ask")
    def ask():
        body = request.get_json(force=True, silent=True) or {}
        q = (body.get("question") or "").strip()[:500]
        if not q:
            return jsonify({"error": "empty question"}), 400
        hits = r.search(q, k=int(body.get("k", 5)), topic=body.get("topic") or None)
        out = answer(q, hits)
        out["hits"] = [{"q": h.item.q, "a": h.item.a, "topic": h.item.topic, "section": h.item.section,
                        "sources": h.item.sources, "ai_written": h.item.answered_by_llm,
                        "score": round(h.score, 4), "keyword_rank": h.rank_keyword,
                        "semantic_rank": h.rank_semantic} for h in hits]
        return jsonify(out)

    return app


def serve(kb_dir: Path, host: str = "127.0.0.1", port: int = 8501) -> None:
    app = create_app(kb_dir)
    print(f"Ask my notes: http://{host}:{port}")
    app.run(host=host, port=port)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Ask my notes</title>
<style>
:root{--bg:#F7F6F2;--card:#fff;--ink:#16181D;--muted:#5C6270;--line:#E3E1DA;--accent:#2340C7;--code:#F1F0EA}
@media (prefers-color-scheme:dark){:root{--bg:#121318;--card:#1B1D24;--ink:#ECEDF0;--muted:#9EA3B0;--line:#2C2F39;--accent:#7F95FF;--code:#23262F}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:820px;margin:0 auto;padding:32px 16px 64px}h1{font-size:28px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 24px}form{display:flex;gap:8px;flex-wrap:wrap}
input,select,button{font:inherit;border-radius:10px;border:1px solid var(--line);padding:10px 12px;background:var(--card);color:var(--ink)}
input{flex:1 1 320px}button{background:var(--accent);color:#fff;border:0;font-weight:600;cursor:pointer}
.answer,.hit{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:16px}
.answer{border-left:4px solid var(--accent);white-space:pre-wrap}.hit h3{font-size:16px;margin:0 0 6px}
.meta{color:var(--muted);font-size:13px;margin-bottom:8px}.a{white-space:pre-wrap;font-size:15px}
pre{background:var(--code);padding:10px;border-radius:8px;overflow:auto;font-size:13px}
.tag{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:99px;padding:1px 8px;margin-right:6px}
</style></head><body><main>
<h1>Ask my notes</h1><p class="sub">Hybrid search (keyword + embeddings) over your knowledge base.
Answers come only from your notes and cite them.</p>
<form id="f"><input id="q" placeholder="e.g. how do I handle late-arriving data in streaming?" autofocus>
<select id="t"><option value="">All topics</option></select><button>Ask</button></form>
<div id="out"></div></main>
<script>
const $=s=>document.querySelector(s);
fetch('/api/topics').then(r=>r.json()).then(ts=>ts.forEach(t=>{const o=document.createElement('option');o.value=o.textContent=t;$('#t').append(o)}));
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function fmt(s){return esc(s).replace(/```(\\w*)\\n?([\\s\\S]*?)```/g,(m,l,c)=>'<pre>'+c+'</pre>')}
$('#f').onsubmit=async e=>{e.preventDefault();const q=$('#q').value.trim();if(!q)return;
 $('#out').innerHTML='<p class="sub">Searching...</p>';
 const r=await fetch('/api/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q,topic:$('#t').value})}).then(r=>r.json());
 let h=r.answer?'<div class="answer">'+fmt(r.answer)+'</div>':'<p class="sub">Top matches from your notes'+(r.generated===false?' (set ANTHROPIC_API_KEY for a written answer)':'')+':</p>';
 (r.hits||[]).forEach((x,i)=>{h+='<div class="hit"><h3>['+(i+1)+'] '+esc(x.q)+'</h3><div class="meta"><span class="tag">'+esc(x.topic)+'</span>'+
  (x.ai_written?'<span class="tag">AI-written</span>':'')+esc(x.sources.join(', '))+'</div><div class="a">'+fmt(x.a||'(no answer yet)')+'</div></div>'});
 $('#out').innerHTML=h||'<p class="sub">Nothing found.</p>'};
</script></body></html>"""
