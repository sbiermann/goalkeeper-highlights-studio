from __future__ import annotations

import gc
import json
import re
import time
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


def calculate_routing(candidate: Candidate, config: dict) -> None:
    cfg = config.get("qwen", {}).get("routing", {})
    if not cfg.get("enabled", True):
        candidate.routing_category = "MEDIUM"
        candidate.routing_score = candidate.event_score
        return

    # Use available features from EventEngine
    # event_score is already a weighted combination of proximity, approach, departure, motion, etc.
    score = candidate.event_score
    
    # Heuristic adjustments
    reasons = []
    
    # Possession bonus for potential catches/distribution
    if candidate.possession_duration > 0.8:
        score += 0.05
        reasons.append("possession")
    
    # Dynamic signal
    if candidate.approach_speed > 0.6 or candidate.direction_change > 0.4:
        score += 0.08
        reasons.append("dynamics")
        
    # Spatial penalty: if too far from goal (if coordinates were available and used here)
    # But event_score already includes proximity.
    
    # Recovery candidates should generally be MEDIUM unless very weak
    if candidate.recovery_candidate and cfg.get("recovery_force_medium", True):
        if score > cfg.get("low_threshold", 0.15):
             candidate.routing_category = "MEDIUM"
             candidate.routing_score = score
             candidate.routing_reason = "recovery_force_medium"
             return

    candidate.routing_score = max(0.0, min(1.0, score))
    
    if candidate.routing_score >= cfg.get("high_threshold", 0.85):
        candidate.routing_category = "HIGH"
    elif candidate.routing_score <= cfg.get("low_threshold", 0.15):
        candidate.routing_category = "LOW"
    else:
        candidate.routing_category = "MEDIUM"
    
    candidate.routing_reason = ",".join(reasons)


def should_retry_qwen(*, confidence: float, contact: bool, parse_failed: bool, candidate: Candidate, config: dict) -> bool:
    routing = config.get("qwen", {}).get("routing", {})
    if not routing.get("retry_enabled", True) or candidate.qwen_retry_count > 0:
        return False
    if parse_failed:
        return True
    low = float(routing.get("retry_min_confidence", 0.40))
    high = float(routing.get("retry_max_confidence", 0.70))
    if low <= confidence <= high:
        return True
    if candidate.recovery_candidate and not contact and confidence >= float(routing.get("recovery_retry_min_confidence", 0.30)):
        return True
    if candidate.contact_frames <= 1 and candidate.keeper_motion >= float(routing.get("short_action_motion_threshold", 0.30)):
        return True
    return False


def classify(video, candidates: list[Candidate], config: dict) -> dict[str, float | int]:
    cfg = config["qwen"]
    if not cfg["enabled"]:
        for candidate in candidates:
            if not candidate.category or candidate.category == "unclassified":
                candidate.category = "heuristic"
            if not candidate.description:
                candidate.description = "Mehrfach bestätigte Ballnähe zum re-identifizierten Torwart."
            candidate.quality_score = max(candidate.quality_score, candidate.heuristic_score)
        return {"heuristic_seconds": 0.0, "qwen_first_pass_seconds": 0.0, "qwen_second_pass_seconds": 0.0}

    minimum_confidence = float(cfg.get("minimum_confidence", 0.55))
    minimum_quality = float(cfg.get("minimum_quality_score", 0.48))
    keep_all = bool(cfg.get("keep_all_candidates", False))

    routing_cfg = cfg.get("routing", {})
    routing_enabled = bool(routing_cfg.get("enabled", True))

    stats: dict[str, float | int] = {"heuristic_seconds": 0.0, "qwen_first_pass_seconds": 0.0, "qwen_second_pass_seconds": 0.0}
    model_loaded = False
    processor = None
    model = None
    device = None

    for index, candidate in enumerate(candidates, 1):
        if not candidate.accepted and not keep_all:
            continue
        
        # Calculate routing
        routing_started = time.perf_counter()
        calculate_routing(candidate, config)
        stats["heuristic_seconds"] = float(stats["heuristic_seconds"]) + (time.perf_counter() - routing_started)
        
        if routing_enabled:
            if candidate.routing_category == "HIGH":
                candidate.accepted = True
                if not candidate.category or candidate.category == "unclassified":
                    candidate.category = "heuristic_high"
                if not candidate.description:
                    candidate.description = "Starke Torwartaktion direkt heuristisch akzeptiert."
                candidate.quality_score = max(candidate.quality_score, candidate.routing_score)
                if bool(config.get("runtime", {}).get("verbose_console", False)):
                    print(f"Candidate {index} HIGH: Directly accepted (score={candidate.routing_score:.2f})")
                continue
            elif candidate.routing_category == "LOW":
                candidate.accepted = False
                candidate.rejection_reason = "heuristic_low_score"
                if bool(config.get("runtime", {}).get("verbose_console", False)):
                    print(f"Candidate {index} LOW: Directly rejected (score={candidate.routing_score:.2f})")
                continue
            # MEDIUM continues to Qwen

        if not model_loaded:
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
            device = next(model.parameters()).device
            model_loaded = True

        if bool(config.get("runtime", {}).get("verbose_console", False)):
            print(f"Qwen {index}/{len(candidates)} at {candidate.trigger_time:.1f}s (category={candidate.routing_category})")
        try:
            candidate.qwen_first_pass_called = True
            first_pass_started = time.perf_counter()
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
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            data = parse_json(raw)
            contact = bool(data.get("goalkeeper_contact", False))
            qwen_confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            qwen_quality = max(0.0, min(1.0, float(data.get("quality_score", qwen_confidence))))
            candidate.qwen_first_pass_seconds = time.perf_counter() - first_pass_started
            stats["qwen_first_pass_seconds"] = float(stats["qwen_first_pass_seconds"]) + candidate.qwen_first_pass_seconds
            first_pass_contact = contact
            first_pass_confidence = qwen_confidence

            # Check if retry is needed
            retry_cfg = cfg.get("routing", {})
            uncertain = should_retry_qwen(
                confidence=qwen_confidence, contact=contact, parse_failed=False, candidate=candidate, config=config
            )

            if uncertain:
                candidate.qwen_retry_count += 1
                candidate.qwen_second_pass_called = True
                second_pass_started = time.perf_counter()
                candidate.qwen_retry_confidence = qwen_confidence
                if bool(config.get("runtime", {}).get("verbose_console", False)):
                    print(f"  Uncertain result (conf={qwen_confidence:.2f}), retrying with more context...")
                
                # Expand context for retry
                orig_start, orig_end = candidate.start, candidate.end
                candidate.start = max(0, candidate.start - float(retry_cfg.get("retry_context_before", 2.0)))
                # We don't have total duration easily here, but extract_frames handles it via video.locate or VideoCapture
                candidate.end = candidate.end + float(retry_cfg.get("retry_context_after", 2.0))
                
                try:
                    retry_images = extract_frames(video, candidate, int(retry_cfg.get("retry_frames", 12)))
                    if len(retry_images) >= 3:
                        retry_content = [{"type": "image", "image": image} for image in retry_images]
                        retry_content.append({
                            "type": "text",
                            "text": (
                                "Dies ist ein zweiter Durchlauf mit mehr zeitlichem Kontext. "
                                "Achte besonders auf kurze Paraden, Fangaktionen oder schnelle Ballkontakte, die im ersten Durchlauf eventuell schwer zu sehen waren. "
                                "Beurteile, ob der Torwart den Ball berührt oder kontrolliert. "
                                "Antworte ausschließlich als JSON: "
                                '{"goalkeeper_contact":true,"category":"save|catch|punch|clearance|pass|goal_kick|throw|dribble|other|none",'
                                '"confidence":0.0,"quality_score":0.0,"description":"kurze deutsche Beschreibung"}'
                            ),
                        })
                        retry_messages = [{"role": "user", "content": retry_content}]
                        retry_text = processor.apply_chat_template(retry_messages, tokenize=False, add_generation_prompt=True)
                        retry_inputs = processor(text=[retry_text], images=retry_images, padding=True, return_tensors="pt")
                        retry_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in retry_inputs.items()}
                        with torch.inference_mode():
                            retry_generated = model.generate(**retry_inputs, max_new_tokens=200, do_sample=False)
                        raw = processor.batch_decode(retry_generated[:, retry_inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                        data = parse_json(raw)
                        contact = bool(data.get("goalkeeper_contact", False))
                        qwen_confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
                        qwen_quality = max(0.0, min(1.0, float(data.get("quality_score", qwen_confidence))))
                        candidate.qwen_second_pass_rescued = (not first_pass_contact) and contact
                except Exception as retry_exc:
                    if bool(config.get("runtime", {}).get("verbose_console", False)):
                        print(f"  Retry failed: {retry_exc}")
                finally:
                    candidate.qwen_second_pass_seconds = time.perf_counter() - second_pass_started
                    stats["qwen_second_pass_seconds"] = float(stats["qwen_second_pass_seconds"]) + candidate.qwen_second_pass_seconds
                    # Restore original boundaries for the candidate object, 
                    # but the classification decision is based on the retry.
                    candidate.start, candidate.end = orig_start, orig_end

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
            if candidate.qwen_first_pass_called and candidate.qwen_first_pass_seconds == 0.0:
                candidate.qwen_first_pass_seconds = time.perf_counter() - first_pass_started
                stats["qwen_first_pass_seconds"] = float(stats["qwen_first_pass_seconds"]) + candidate.qwen_first_pass_seconds
            candidate.category = "qwen_error"
            candidate.description = f"Qwen-Fehler; heuristische Entscheidung bleibt erhalten: {exc}"
            candidate.quality_score = candidate.heuristic_score
            candidate.qwen_raw = str(exc)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return stats
