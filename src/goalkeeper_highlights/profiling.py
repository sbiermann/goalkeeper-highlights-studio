from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


@dataclass(slots=True)
class ProfileSample:
    wall_seconds: float
    video_seconds: float
    frame_index: int
    processed_frames: int
    realtime_factor: float
    effective_fps: float
    loop_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    candidate_ms: float
    database_ms: float
    preview_ms: float
    raw_candidates: int
    detections: int
    persons: int
    balls: int
    cpu_percent: float
    ram_used_gb: float
    ram_percent: float
    process_ram_mb: float
    gpu_percent: float | None
    gpu_memory_used_mb: float | None
    gpu_memory_total_mb: float | None
    gpu_temperature_c: float | None


class PerformanceProfiler:
    """Collects low-overhead runtime metrics and writes CSV, JSON and HTML reports."""

    def __init__(self, output_dir: Path, enabled: bool = True, sample_interval_seconds: float = 5.0, store=None):
        self.enabled = enabled
        self.output_dir = output_dir
        self.sample_interval_seconds = max(0.5, float(sample_interval_seconds))
        self.started = time.perf_counter()
        self.last_sample = self.started - self.sample_interval_seconds
        self.samples: list[ProfileSample] = []
        self.store = store
        self.process = psutil.Process()
        self.process.cpu_percent(None)
        psutil.cpu_percent(None)
        self._nvml: Any = None
        self._gpu_handle: Any = None
        self.gpu_name: str | None = None
        if enabled:
            self._init_gpu()

    def _init_gpu(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_name = pynvml.nvmlDeviceGetName(self._gpu_handle)
            self.gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        except Exception:
            self._nvml = None
            self._gpu_handle = None

    def should_sample(self) -> bool:
        return self.enabled and time.perf_counter() - self.last_sample >= self.sample_interval_seconds

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            value = float(value)
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    def sample(
        self,
        *,
        video_seconds: float,
        frame_index: int,
        processed_frames: int,
        loop_ms: float,
        speed: dict[str, Any] | None,
        candidate_ms: float,
        database_ms: float,
        preview_ms: float,
        raw_candidates: int,
        detections: int,
        persons: int,
        balls: int,
    ) -> ProfileSample | None:
        if not self.should_sample():
            return None
        now = time.perf_counter()
        wall = now - self.started
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info().rss / (1024 * 1024)
        gpu_percent = gpu_used = gpu_total = gpu_temp = None
        if self._nvml is not None and self._gpu_handle is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_percent = float(util.gpu)
                gpu_used = mem.used / (1024 * 1024)
                gpu_total = mem.total / (1024 * 1024)
                gpu_temp = float(self._nvml.nvmlDeviceGetTemperature(self._gpu_handle, self._nvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
        speed = speed or {}
        item = ProfileSample(
            wall_seconds=wall,
            video_seconds=video_seconds,
            frame_index=frame_index,
            processed_frames=processed_frames,
            realtime_factor=video_seconds / max(wall, 0.001),
            effective_fps=processed_frames / max(wall, 0.001),
            loop_ms=loop_ms,
            preprocess_ms=self._safe_float(speed.get("preprocess")),
            inference_ms=self._safe_float(speed.get("inference")),
            postprocess_ms=self._safe_float(speed.get("postprocess")),
            candidate_ms=candidate_ms,
            database_ms=database_ms,
            preview_ms=preview_ms,
            raw_candidates=raw_candidates,
            detections=detections,
            persons=persons,
            balls=balls,
            cpu_percent=float(psutil.cpu_percent(None)),
            ram_used_gb=memory.used / (1024 ** 3),
            ram_percent=float(memory.percent),
            process_ram_mb=process_memory,
            gpu_percent=gpu_percent,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
            gpu_temperature_c=gpu_temp,
        )
        self.samples.append(item)
        if self.store is not None:
            self.store.append_profile_sample(asdict(item))
        self.last_sample = now
        return item

    @staticmethod
    def format_console(sample: ProfileSample) -> str:
        gpu = "GPU n/a"
        if sample.gpu_percent is not None:
            gpu = f"GPU {sample.gpu_percent:.0f}%"
            if sample.gpu_memory_used_mb is not None:
                gpu += f", VRAM {sample.gpu_memory_used_mb / 1024:.1f} GB"
        return (
            f"profile: {sample.effective_fps:.1f} FPS, {sample.realtime_factor:.2f}x, "
            f"infer {sample.inference_ms:.1f} ms, pre {sample.preprocess_ms:.1f} ms, "
            f"post {sample.postprocess_ms:.1f} ms, loop {sample.loop_ms:.1f} ms, "
            f"CPU {sample.cpu_percent:.0f}%, RAM {sample.process_ram_mb:.0f} MB, {gpu}"
        )

    def finish(self, extra: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        target = self.output_dir / "profiling"
        target.mkdir(parents=True, exist_ok=True)
        rows = [asdict(item) for item in self.samples]
        if rows:
            with (target / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = self._summary(extra or {})
        (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (target / "report.html").write_text(self._html(summary, rows), encoding="utf-8")
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _summary(self, extra: dict[str, Any]) -> dict[str, Any]:
        def avg(name: str) -> float | None:
            values = [getattr(s, name) for s in self.samples if getattr(s, name) is not None]
            return round(statistics.fmean(values), 3) if values else None

        def peak(name: str) -> float | None:
            values = [getattr(s, name) for s in self.samples if getattr(s, name) is not None]
            return round(max(values), 3) if values else None

        result: dict[str, Any] = {
            "system": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_logical_cores": psutil.cpu_count(),
                "gpu": self.gpu_name,
            },
            "samples": len(self.samples),
            "averages": {
                "realtime_factor": avg("realtime_factor"),
                "effective_fps": avg("effective_fps"),
                "preprocess_ms": avg("preprocess_ms"),
                "inference_ms": avg("inference_ms"),
                "postprocess_ms": avg("postprocess_ms"),
                "loop_ms": avg("loop_ms"),
                "candidate_ms": avg("candidate_ms"),
                "database_ms": avg("database_ms"),
                "preview_ms": avg("preview_ms"),
                "cpu_percent": avg("cpu_percent"),
                "process_ram_mb": avg("process_ram_mb"),
                "gpu_percent": avg("gpu_percent"),
                "gpu_memory_used_mb": avg("gpu_memory_used_mb"),
            },
            "peaks": {
                "process_ram_mb": peak("process_ram_mb"),
                "ram_percent": peak("ram_percent"),
                "gpu_percent": peak("gpu_percent"),
                "gpu_memory_used_mb": peak("gpu_memory_used_mb"),
                "gpu_temperature_c": peak("gpu_temperature_c"),
            },
        }
        result.update(extra)
        return result

    @staticmethod
    def _html(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
        payload = json.dumps(rows, ensure_ascii=False)
        summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
        return f"""<!doctype html>
<html lang='de'><head><meta charset='utf-8'><title>Performance-Profiling</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#e6edf3;max-width:1200px;margin:30px auto;padding:0 20px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}}canvas{{width:100%;height:260px;background:#161b22;border:1px solid #30363d;border-radius:10px;margin:16px 0}}pre{{background:#161b22;padding:16px;overflow:auto;border-radius:10px}}</style></head>
<body><h1>Performance-Profiling</h1><div class='cards' id='cards'></div>
<h2>Verlauf</h2><canvas id='chart' width='1100' height='280'></canvas><h2>Zusammenfassung</h2><pre>{summary_json}</pre>
<script>
const rows={payload}; const avg={json.dumps(summary.get('averages', {}), ensure_ascii=False)};
const cards=document.getElementById('cards');
for(const [k,v] of Object.entries(avg)){{cards.innerHTML+=`<div class='card'><strong>${{k}}</strong><br>${{v ?? 'n/a'}}</div>`;}}
const c=document.getElementById('chart'),x=c.getContext('2d'); x.clearRect(0,0,c.width,c.height);
const series=[['GPU %','gpu_percent','#58a6ff'],['CPU %','cpu_percent','#3fb950'],['Realtime x10','realtime_factor','#f0883e']];
function px(i){{return 45+i*(c.width-70)/Math.max(1,rows.length-1)}} function py(v){{return c.height-30-(v||0)*(c.height-55)/100}}
x.strokeStyle='#30363d'; for(let v=0;v<=100;v+=20){{x.beginPath();x.moveTo(40,py(v));x.lineTo(c.width-20,py(v));x.stroke();}}
for(const [name,key,color] of series){{x.strokeStyle=color;x.lineWidth=2;x.beginPath(); rows.forEach((r,i)=>{{let v=r[key]; if(key==='realtime_factor')v*=10; i?x.lineTo(px(i),py(v)):x.moveTo(px(i),py(v));}});x.stroke();}}
</script></body></html>"""
