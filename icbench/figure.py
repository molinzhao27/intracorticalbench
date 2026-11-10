"""WER against number of labelled trials for one sentence set, as a self-contained HTML/SVG chart.

  uv run python -m icbench.figure            # writes runs/figure.html
"""
from __future__ import annotations

import html
import json
import sys

import numpy as np

from .models import ROOT

NAMES = {"astra": "GPT Astra", "fable": "Fable 5.2", "opus": "Opus 5.5"}
COLOR = {"astra": "var(--chart-1, #2a78d6)", "fable": "var(--chart-2, #eb6834)", "opus": "var(--chart-3, #1baf7a)"}
KS = [0, 16, 64, 192]
W, H, L, R, T, B = 728, 430, 46, 190, 14, 44


def main(dataset: str = "t15_small") -> None:
    rep = json.loads((ROOT / "runs/report.json").read_text())
    base = json.loads((ROOT / "runs/baselines.json").read_text())
    rows = [r for r in rep["sentences"] if r["dataset"] == dataset and r["fmt"] == "text"]
    x = lambda k: L + KS.index(k) * (W - L - R) / (len(KS) - 1)
    y = lambda v: T + (1.3 - v) * (H - T - B) / 1.3
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Word error rate against number of labelled trials">']
    for v in np.arange(0, 1.31, 0.2):
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    for k in KS:
        s.append(f'<text x="{x(k):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{k}</text>')
    s.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 6}" class="axis" text-anchor="middle">labelled trials in the prompt</text>')
    s.append(f'<text x="12" y="{(T + H - B) / 2:.1f}" class="axis" text-anchor="middle" transform="rotate(-90 12 {(T + H - B) / 2:.1f})">word error rate</text>')

    # references: activity withheld (mean over models and efforts) and a small trained decoder
    wh = [float(np.mean([r["withheld_wer"] for r in rows if r["k"] == k and "withheld_wer" in r])) for k in KS]
    s.append('<polyline class="ref" points="' + " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in zip(KS, wh)) + '"/>')
    s.append(f'<text x="{x(64) + 40:.1f}" y="{y(wh[2]) - 30:.1f}" class="lab muted">activity withheld (guessing)</text>'
             f'<line x1="{x(64) + 36:.1f}" y1="{y(wh[2]) - 28:.1f}" x2="{x(64) + 14:.1f}" y2="{y(wh[2]) - 3:.1f}" class="lead"/>')
    gru = {b["k"]: b["score"]["wer"] for b in base["sentences"] if b["dataset"] == dataset and b["decoder"] == "GRU-CTC, characters"}
    s.append('<polyline class="ref dot" points="' + " ".join(f"{x(k):.1f},{y(gru[k]):.1f}" for k in KS[1:]) + '"/>')
    ends = [(gru[192], "GRU-CTC on the same trials", "muted", None)]

    pts = []
    for m in ("opus", "fable", "astra"):
        for eff in ("low", "high"):
            cell = sorted((r for r in rows if r["model"] == m and r["effort"] == eff), key=lambda r: r["k"])
            if not cell:
                continue
            dash = ' stroke-dasharray="5 4"' if eff == "low" else ""
            s.append(f'<polyline fill="none" stroke="{COLOR[m]}" stroke-width="2"{dash} points="' + " ".join(f"{x(r['k']):.1f},{y(r['wer']):.1f}" for r in cell) + '"/>')
            pts += [(m, eff, r) for r in cell]
            last = cell[-1]
            label = f"{NAMES[m]}, {eff} effort"
            if last["k"] == 192:
                ends.append((last["wer"], label, "", m))
            else:   # line stops early: label it at its last point
                note = " (no answer at 192)" if m == "fable" else ""
                anchor, dx = ("start", 10) if last["k"] == 16 else ("end", -10)
                s.append(f'<text x="{x(last["k"]) + dx:.1f}" y="{y(last["wer"]) + 16:.1f}" class="lab" text-anchor="{anchor}">{label}{note}</text>')
    ends.sort(key=lambda e: -e[0])
    ly = None
    for v, label, cls, m in ends:                                  # right-edge labels, pushed apart
        yy = y(v) + 4 if ly is None else max(y(v) + 4, ly + 15)
        ly = yy
        mark = f'<rect x="{W - R + 10}" y="{yy - 9:.1f}" width="10" height="10" rx="2" fill="{COLOR[m]}"/>' if m else ""
        s.append(f'{mark}<text x="{W - R + (26 if m else 10)}" y="{yy:.1f}" class="lab {cls}">{label}</text>')
    for m, eff, r in pts:
        tip = (f"{NAMES[m]}, {eff} effort, {r['k']} labelled trials|WER {r['wer']:.2f} · chance {r['chance']:.2f} (p = {r['p']:.4f})|"
               f"activity withheld {r.get('withheld_wer', float('nan')):.2f} · exact sentences {r['exact'] * 100:.0f}% · answered {r['answered']}/{r['n']}")
        fill = COLOR[m] if eff == "high" else "var(--background, #fff)"
        s.append(f'<circle cx="{x(r["k"]):.1f}" cy="{y(r["wer"]):.1f}" r="4" fill="{fill}" stroke="{COLOR[m]}" stroke-width="2"/>'
                 f'<circle cx="{x(r["k"]):.1f}" cy="{y(r["wer"]):.1f}" r="13" fill="transparent" class="hit" data-tip="{html.escape(tip)}"/>')
    s.append("</svg>")

    tr = "".join(f"<tr><td>{NAMES[m]}</td><td>{eff}</td><td>{r['k']}</td><td>{r['wer']:.2f}</td><td>{r['chance']:.2f}</td><td>{r['p']:.4f}</td>"
                 f"<td>{r.get('withheld_wer', float('nan')):.2f}</td><td>{r['exact'] * 100:.0f}%</td><td>{r['answered']}/{r['n']}</td></tr>"
                 for m, eff, r in sorted(pts, key=lambda p: (p[0], p[1], p[2]["k"])))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
.wrap{{position:relative;font-family:var(--font-sans,system-ui);color:var(--foreground,#0b0b0b)}}
svg{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--border,#e1e0d9);stroke-width:1}} .tick{{font-size:11px;fill:var(--muted-foreground,#898781)}}
.axis{{font-size:12px;fill:var(--muted-foreground,#898781)}} .lab{{font-size:12px;fill:var(--foreground,#0b0b0b)}}
.lab.muted{{fill:var(--muted-foreground,#898781)}} .ref{{fill:none;stroke:var(--muted-foreground,#898781);stroke-width:1.5}}
.ref.dot{{stroke-dasharray:2 4}} .lead{{stroke:var(--muted-foreground,#898781);stroke-width:1}}
.key{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12px;color:var(--muted-foreground,#898781);margin:2px 0 8px {L}px}}
.key i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.key svg{{width:26px;height:8px;display:inline-block;margin-right:6px;vertical-align:1px}}
#tip{{position:absolute;pointer-events:none;display:none;background:var(--popover,#fff);color:var(--popover-foreground,#0b0b0b);border:1px solid var(--border,#e1e0d9);border-radius:var(--radius,6px);padding:8px 10px;font-size:12px;line-height:1.45;max-width:320px;box-shadow:0 2px 8px rgba(0,0,0,.12)}}
#tip b{{display:block}} details{{font-size:12px;margin-top:6px}} summary{{cursor:pointer;color:var(--muted-foreground,#898781)}}
table{{border-collapse:collapse;margin-top:6px;width:100%}} td,th{{text-align:left;padding:3px 8px;border-bottom:1px solid var(--border,#e1e0d9)}}
</style></head><body><div class="wrap">
<div class="key"><span><i style="background:{COLOR['astra']}"></i>GPT Astra</span><span><i style="background:{COLOR['fable']}"></i>Fable 5.2</span><span><i style="background:{COLOR['opus']}"></i>Opus 5.5</span>
<span><svg viewBox="0 0 26 8"><line x1="0" y1="4" x2="26" y2="4" stroke="currentColor" stroke-width="2"/></svg>high effort</span>
<span><svg viewBox="0 0 26 8"><line x1="0" y1="4" x2="26" y2="4" stroke="currentColor" stroke-width="2" stroke-dasharray="5 4"/></svg>low effort</span></div>
{''.join(s)}
<div id="tip"></div>
<details><summary>Values as a table</summary><table><tr><th>model</th><th>effort</th><th>labelled trials</th><th>WER</th><th>chance</th><th>p</th><th>activity withheld</th><th>exact</th><th>answered</th></tr>{tr}</table></details>
</div><script>
const tip=document.getElementById('tip'),wrap=document.querySelector('.wrap');
document.querySelectorAll('.hit').forEach(c=>{{
  c.addEventListener('mousemove',e=>{{const [a,b,d]=c.dataset.tip.split('|');tip.innerHTML='<b>'+a+'</b>'+b+'<br>'+d;tip.style.display='block';
    const r=wrap.getBoundingClientRect();let x=e.clientX-r.left+14,y=e.clientY-r.top+14;if(x+330>r.width)x=Math.max(0,e.clientX-r.left-334);tip.style.left=x+'px';tip.style.top=y+'px';}});
  c.addEventListener('mouseleave',()=>tip.style.display='none');}});
</script></body></html>"""
    (ROOT / "runs/figure.html").write_text(page)
    print("wrote runs/figure.html", len(page), "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])
