from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

from .models import Candidate


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "–"
    value = float(seconds)
    if value < 60:
        return f"{value:.1f}s"
    return f"{int(value // 60)}:{int(value % 60):02d} min"


def _score(value: float) -> str:
    return f"{max(0.0, min(1.0, value))*100:.0f}%"


def _breakdown(candidate: Candidate) -> str:
    if not candidate.score_breakdown:
        return ""
    rows = []
    for name, value in candidate.score_breakdown.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        sign = "+" if value >= 0 else "−"
        label = name.replace("_", " ").title()
        rows.append(f"<li><span>{sign} {html.escape(label)}</span><b>{abs(value):.3f}</b></li>")
    if not rows:
        return ""
    return "<details><summary>Score-Aufschlüsselung</summary><ul class=breakdown>" + "".join(rows) + "</ul></details>"


def _video(candidate: Candidate, output: Path) -> str:
    if not candidate.clip_path:
        return ""
    try:
        src = Path(candidate.clip_path).relative_to(output).as_posix()
    except ValueError:
        src = Path(candidate.clip_path).as_posix()
    return f'<video controls preload=metadata src="{html.escape(src)}"></video>'


def _write_analysis(output: Path, candidates: list[Candidate], timings: dict | None = None, keeper_detection: dict | None = None) -> None:
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    with (analysis / "keeper_tracks.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trigger_time", "keeper_label", "bytetrack_id", "identity_confidence", "accepted"])
        for c in candidates:
            writer.writerow([c.trigger_time, c.keeper_label, c.keeper_track_id, c.identity_confidence, c.accepted])
    values = [c.event_score for c in candidates]
    max_time = max((x.trigger_time for x in candidates), default=1.0)
    bars = "".join(
        f'<div class="event {"accepted" if c.accepted else "rejected"}" style="left:{min(99,c.trigger_time/max(1,max_time)*100):.2f}%;height:{20+70*c.event_score:.1f}%" title="{_fmt(c.trigger_time)} {html.escape(c.category)} {_score(c.event_score)}"></div>'
        for c in candidates
    )
    (analysis / "timeline.html").write_text(
        f"""<!doctype html><meta charset=utf-8><title>Event Timeline</title><style>body{{font-family:system-ui;background:#0d1117;color:#eee;padding:30px}}.chart{{height:260px;border-bottom:1px solid #777;position:relative}}.event{{position:absolute;bottom:0;width:8px;background:#58a6ff}}.rejected{{background:#f85149}}</style><h1>Event Timeline</h1><div class=chart>{bars}</div><p>{len(candidates)} Kandidaten</p>""",
        encoding="utf-8",
    )
    bins = [0] * 10
    for value in values:
        bins[min(9, int(value * 10))] += 1
    hist = "".join(f"<tr><td>{i*10}–{i*10+9}%</td><td>{n}</td></tr>" for i, n in enumerate(bins))
    (analysis / "score_histogram.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>Score Histogram</title><style>body{{font-family:system-ui;background:#0d1117;color:#eee;padding:30px}}td{{padding:8px 20px;border-bottom:1px solid #333}}</style><h1>Score-Verteilung</h1><table>{hist}</table>",
        encoding="utf-8",
    )
    if timings is not None:
        (analysis / "performance.json").write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")
    if keeper_detection:
        (analysis / "goalkeeper_detection.json").write_text(json.dumps(keeper_detection, indent=2, ensure_ascii=False), encoding="utf-8")
        weights = keeper_detection.get("weights", {})
        rows = "".join(
            f"<tr><td>{i}</td><td>{html.escape(str(row.get('track_id', '–')))}</td><td>{_score(float(row.get('score', 0)))}</td><td>{_score(float(row.get('shirt_uniqueness', 0)))}</td><td>{float(row.get('shirt_uniqueness', 0))*float(weights.get('shirt_uniqueness', 0)):+.3f}</td><td>{_score(float(row.get('camera_proximity', 0)))}</td><td>{float(row.get('camera_proximity', 0))*float(weights.get('camera_proximity', 0)):+.3f}</td><td>{_score(float(row.get('goal_area', 0)))}</td><td>{float(row.get('goal_area', 0))*float(weights.get('goal_area', 0)):+.3f}</td><td>{_score(float(row.get('low_movement', 0)))}</td><td>{float(row.get('low_movement', 0))*float(weights.get('low_movement', 0)):+.3f}</td><td>{_score(float(row.get('ball_contact', 0)))}</td><td>{float(row.get('ball_contact', 0))*float(weights.get('ball_contact', 0)):+.3f}</td></tr>"
            for i, row in enumerate(keeper_detection.get("ranking", []), 1)
        )
        selected = html.escape(str(keeper_detection.get("selected_track_id", "–")))
        confidence = _score(float(keeper_detection.get("confidence", 0)))
        initial_confidence = _score(float(keeper_detection.get("initial_confidence", keeper_detection.get("confidence", 0))))
        stabilized_confidence = _score(float(keeper_detection.get("stabilized_confidence", keeper_detection.get("confidence", 0))))
        reid_count = int(keeper_detection.get("reidentification_count", 0))
        method = html.escape(str(keeper_detection.get("method", "unknown")))
        page = f"""<!doctype html><html lang=de><meta charset=utf-8><title>Automatische Torwarterkennung</title><style>body{{font-family:system-ui;background:#0d1117;color:#eee;padding:30px}}.card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;max-width:1400px;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #30363d;text-align:left;white-space:nowrap}}</style><div class=card><h1>Automatische Torwarterkennung</h1><p><b>Keeper #1:</b> ByteTrack {selected} · Startkonfidenz {initial_confidence} · stabilisierte Konfidenz {stabilized_confidence} · Re-Identifikationen {reid_count} · Methode {method}</p><p>Die fachliche Identität <b>Keeper #1</b> bleibt auch bei späteren ByteTrack-Wechseln stabil. Die Spalten „Beitrag“ zeigen den gewichteten Anteil am Gesamtscore.</p><table><thead><tr><th>Rang</th><th>ByteTrack</th><th>Gesamt</th><th>Trikot</th><th>Beitrag</th><th>Kameranähe</th><th>Beitrag</th><th>Torbereich</th><th>Beitrag</th><th>Bewegung</th><th>Beitrag</th><th>Ballkontakt</th><th>Beitrag</th></tr></thead><tbody>{rows}</tbody></table></div></html>"""
        (analysis / "goalkeeper_detection.html").write_text(page, encoding="utf-8")


def write_reports(output: Path, candidates: list[Candidate], timings: dict | None = None, keeper_detection: dict | None = None) -> None:
    data = [c.as_dict() for c in candidates]
    (output / "events.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "events.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        fieldnames = sorted({k for item in data for k in item}) if data else ["start", "end", "category", "accepted"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    accepted = [c for c in candidates if c.accepted]
    rejected = [c for c in candidates if not c.accepted]
    counts = Counter(c.category or "unclassified" for c in accepted)
    reject_counts = Counter(c.rejection_reason or "gefiltert" for c in rejected)

    def card(c: Candidate, rejected_card: bool = False) -> str:
        state = "rejected-event" if rejected_card else "event"
        reason = f'<p class="reason">Ablehnung: {html.escape(c.rejection_reason or "gefiltert")}</p>' if rejected_card else ""
        
        merge_info = ""
        if c.merged_from:
            merge_info = f'<span>Merge: {len(c.merged_from)} Kandidaten ({html.escape(c.merged_reason)})</span>'
        
        clip_end_info = f'<span>Clip-Ende: {html.escape(c.clip_end_reason)}</span>'
        interaction_info = f'<span>Interaktions-Score: {_score(c.interaction_score)}</span>'
        
        return (
            f'<article class="{state}"><div><strong>{html.escape(c.category or "Torwartaktion")}</strong><span>{_fmt(c.start)}–{_fmt(c.end)}</span></div>'
            f'<div class="scores"><span>{html.escape(c.keeper_label)}</span><span>Qualität {_score(c.quality_score)}</span><span>Event {_score(c.event_score)}</span><span>Schwelle {_score(c.acceptance_threshold)}</span><span>Identität {_score(c.identity_confidence)}</span><span>Ball {_score(c.ball_confidence)}</span></div>'
            f'<div class="scores"><span>Kontaktframes {c.contact_frames}</span><span>Kontrolle {c.possession_duration:.2f}s</span><span>Besitzbonus +{c.possession_bonus:.2f}</span><span>Anflug {c.approach_speed:.2f}</span><span>Abflug {c.departure_speed:.2f}</span><span>Richtungswechsel {_score(c.direction_change)}</span><span>Torwartbewegung {c.keeper_motion:.2f}</span></div>'
            f'<div class="scores">{interaction_info} {clip_end_info} {merge_info}</div>'
            f'<p>{html.escape(c.description or "Automatisch erkannte Szene")}</p>{reason}{_breakdown(c)}{_video(c, output)}</article>'
        )

    cards = "".join(card(c) for c in accepted) or "<p>Keine akzeptierten Szenen.</p>"
    rejected_cards = "".join(card(c, True) for c in rejected) or "<p>Keine verworfenen Kandidaten.</p>"
    stats = "".join(f"<li><b>{html.escape(k)}</b><span>{v}</span></li>" for k, v in counts.items())
    reasons = "".join(f"<li><b>{html.escape(k)}</b><span>{v}</span></li>" for k, v in reject_counts.items())
    avg_quality = sum(c.quality_score for c in accepted) / len(accepted) if accepted else 0.0
    best = max((c.quality_score for c in accepted), default=0.0)
    worst = min((c.quality_score for c in accepted), default=0.0)
    avg_possession = sum(c.possession_duration for c in accepted) / len(accepted) if accepted else 0.0
    timings = timings or {}
    perf = "".join([
        f'<li><b>Analyse</b><span>{_duration(timings.get("analysis_seconds"))}</span></li>',
        f'<li><b>Clip-Erstellung</b><span>{_duration(timings.get("clip_creation_seconds"))}</span></li>',
        f'<li><b>Zusammenfügen</b><span>{_duration(timings.get("concat_seconds"))}</span></li>',
        f'<li><b>Encoder</b><span>{html.escape(str(timings.get("encoder", "–")))}</span></li>',
        f'<li><b>Clip-Modus</b><span>{html.escape(str(timings.get("clip_mode", "–")))}</span></li>',
        f'<li><b>Parallele Jobs</b><span>{html.escape(str(timings.get("parallel_jobs", "–")))}</span></li>',
    ])
    keeper_detection = keeper_detection or {}
    keeper_summary = ""
    if keeper_detection:
        keeper_summary = f'<h2>Torwarterkennung</h2><ul><li><b>Identität</b><span>{html.escape(str(keeper_detection.get("keeper_label", "Keeper #1")))}</span></li><li><b>Initialer ByteTrack</b><span>{html.escape(str(keeper_detection.get("selected_track_id", "–")))}</span></li><li><b>Konfidenz</b><span>{_score(float(keeper_detection.get("confidence", 0)))}</span></li><li><b>Methode</b><span>{html.escape(str(keeper_detection.get("method", "–")))}</span></li></ul><p><a href="analysis/goalkeeper_detection.html" style="color:#58a6ff">Details zur automatischen Erkennung</a></p>'
    document = f"""<!doctype html><html lang=de><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>Goalkeeper Highlights Report</title><style>
body{{font-family:Inter,system-ui,sans-serif;margin:0;background:#0d1117;color:#e6edf3}}header{{padding:32px;background:linear-gradient(135deg,#13233a,#0d1117)}}main{{max-width:1200px;margin:auto;padding:24px}}h1{{margin:0 0 8px}}.grid{{display:grid;grid-template-columns:330px 1fr;gap:24px}}.panel,.event,.rejected-event{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:18px}}.rejected-event{{border-color:#7d3838}}ul{{padding:0;list-style:none}}li,.event>div,.rejected-event>div{{display:flex;justify-content:space-between;gap:16px;margin:10px 0}}.scores{{font-size:.88rem;color:#9fb3c8;flex-wrap:wrap;justify-content:flex-start!important}}.breakdown li{{border-bottom:1px solid #30363d;padding:5px}}details{{margin:12px 0}}video{{width:100%;border-radius:10px;margin-top:10px}}.events{{display:grid;gap:16px}}.reason{{color:#ff9b96}}.rejected-section{{margin-top:32px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Goalkeeper Highlights 0.13.11</h1><p>{len(accepted)} akzeptierte Szenen aus {len(candidates)} Kandidaten · Durchschnittsqualität {_score(avg_quality)}</p></header><main><div class=grid><aside class=panel><h2>Statistik</h2><ul>{stats}<li><b>Verworfen</b><span>{len(rejected)}</span></li><li><b>Ø Ballbesitz</b><span>{avg_possession:.2f}s</span></li><li><b>Beste Szene</b><span>{_score(best)}</span></li><li><b>Schwächste akzeptierte</b><span>{_score(worst)}</span></li></ul><h2>Ablehnungsgründe</h2><ul>{reasons or '<li><span>Keine</span></li>'}</ul>{keeper_summary}<h2>Performance</h2><ul>{perf}</ul></aside><section class=events>{cards}</section></div><section class=rejected-section><h2>Verworfene Kandidaten</h2><div class=events>{rejected_cards}</div></section></main></body></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")
    _write_analysis(output, candidates, timings, keeper_detection)
