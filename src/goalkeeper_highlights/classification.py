from __future__ import annotations

import gc
import json
import re
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .models import Candidate


def extract_frames(video, candidate: Candidate, count: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    current_path = None
    cap = None
    try:
        for timestamp in np.linspace(candidate.start, candidate.end, max(3, count)):
            if hasattr(video, "locate"):
                item, local_timestamp = video.locate(float(timestamp))
                path = item.path
            else:
                path = str(video)
                local_timestamp = float(timestamp)
            if path != current_path:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(str(path))
                current_path = path
            cap.set(cv2.CAP_PROP_POS_MSEC, local_timestamp * 1000)
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        if cap is not None:
            cap.release()
    return frames


def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("Qwen returned no JSON object")
    return json.loads(match.group(0))


def _combine_quality(heuristic: float, qwen_confidence: float, contact: bool) -> float:
    semantic = qwen_confidence if contact else (1.0 - qwen_confidence)
    return max(0.0, min(1.0, 0.45 * heuristic + 0.55 * semantic))


def classify(video, candidates: list[Candidate], config: dict) -> None:
    cfg = config["qwen"]
    if not cfg["enabled"]:
        for candidate in candidates:
            if not candidate.category or candidate.category == "unclassified":
                candidate.category = "heuristic"
            if not candidate.description:
                candidate.description = "Mehrfach bestätigte Ballnähe zum re-identifizierten Torwart."
            candidate.quality_score = max(candidate.quality_score, candidate.heuristic_score)
        return

    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available() and cfg.get("use_4bit", True):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    processor = AutoProcessor.from_pretrained(cfg["model"], min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg["model"], **kwargs).eval()

    minimum_confidence = float(cfg.get("minimum_confidence", 0.55))
    minimum_quality = float(cfg.get("minimum_quality_score", 0.48))
    keep_all = bool(cfg.get("keep_all_candidates", False))

    for index, candidate in enumerate(candidates, 1):
        if not candidate.accepted and not keep_all:
            continue
        if bool(config.get("runtime", {}).get("verbose_console", False)):
            print(f"Qwen {index}/{len(candidates)} at {candidate.trigger_time:.1f}s")
        try:
            images = extract_frames(video, candidate, int(cfg["frames_per_candidate"]))
            if len(images) < 3:
                raise RuntimeError("Zu wenige Frames aus dem Kandidatenclip gelesen")
            content = [{"type": "image", "image": image} for image in images]
            content.append({
                "type": "text",
                "text": (
                    "Die Bilder sind chronologisch und stammen von einer festen Kamera hinter dem Tor. "
                    "Beurteile ausschließlich, ob der Torwart aktiv mit dem Ball interagiert. "
                    "Eine bloße Ballnähe, ein Spieler vor dem Torwart oder ein vorbeifliegender Ball gelten NICHT als Kontakt. "
                    "Erlaubte Kategorien: save, catch, punch, clearance, pass, goal_kick, throw, dribble, other, none. "
                    "Gib außerdem einen quality_score für die Eignung als Torwart-Highlight an. "
                    "Antworte ausschließlich als JSON: "
                    '{"goalkeeper_contact":true,"category":"save|catch|punch|clearance|pass|goal_kick|throw|dribble|other|none",'
                    '"confidence":0.0,"quality_score":0.0,"description":"kurze deutsche Beschreibung"}'
                ),
            })
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            data = parse_json(raw)
            contact = bool(data.get("goalkeeper_contact", False))
            qwen_confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            qwen_quality = max(0.0, min(1.0, float(data.get("quality_score", qwen_confidence))))
            candidate.category = str(data.get("category", "other" if contact else "none"))
            candidate.confidence = qwen_confidence
            candidate.description = str(data.get("description", ""))
            candidate.qwen_raw = raw
            candidate.quality_score = max(0.0, min(1.0, 0.40 * candidate.heuristic_score + 0.60 * qwen_quality))
            candidate.accepted = keep_all or (contact and qwen_confidence >= minimum_confidence and candidate.quality_score >= minimum_quality)
            if not candidate.accepted:
                reasons = []
                if not contact:
                    reasons.append("qwen_no_goalkeeper_contact")
                if qwen_confidence < minimum_confidence:
                    reasons.append("qwen_confidence_below_threshold")
                if candidate.quality_score < minimum_quality:
                    reasons.append("quality_score_below_threshold")
                candidate.rejection_reason = ",".join(reasons)
        except Exception as exc:
            candidate.category = "qwen_error"
            candidate.description = f"Qwen-Fehler; heuristische Entscheidung bleibt erhalten: {exc}"
            candidate.quality_score = candidate.heuristic_score
            candidate.qwen_raw = str(exc)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
