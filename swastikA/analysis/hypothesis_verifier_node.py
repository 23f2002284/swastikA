import asyncio
from typing import List, Optional, Any
import os
import json
import re
from google.genai import Client, types
from dotenv import load_dotenv

from swastikA.analysis.schemas import (
    Hypothesis,
    HypothesisStatus,
    Evidence,
    EvidenceType,
    VerificationResult,
    FinalAnalysisSchema
)
from swastikA.analysis.prompts import (
    KOLAM_VERIFICATION_SYSTEM_PROMPT,
    SINGLE_HYPOTHESIS_VERIFICATION_USER_PROMPT
)

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT")
LOCATION = os.getenv("GCP_LOCATION")
INCLUDE_THOUGHTS = True


class KolamHypothesisVerifier:
    """
    Verifies hypotheses about Kolam patterns using Gemini with optional chain-of-thought (thought segments).
    Thought segments are NOT returned verbatim to users; they are summarized into the reasoning field if needed.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro", include_thoughts: bool = INCLUDE_THOUGHTS):
        self.model_name = model_name
        self.system_prompt = KOLAM_VERIFICATION_SYSTEM_PROMPT
        self.include_thoughts = include_thoughts
        self._client: Optional[Client] = Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    async def verify_hypotheses(
        self,
        analysis: FinalAnalysisSchema,
        max_parallel_verifications: int = 3
    ) -> FinalAnalysisSchema:
        if not analysis.extracted_features:
            return analysis

        verification_results: List[VerificationResult] = []
        for i in range(0, len(analysis.extracted_features), max_parallel_verifications):
            batch = analysis.extracted_features[i:i + max_parallel_verifications]
            batch_results = await self._verify_batch(batch, analysis.image_path)
            verification_results.extend(batch_results)

        updated = {vr.hypothesis.id: vr.hypothesis for vr in verification_results}
        analysis.extracted_features = [updated.get(h.id, h) for h in analysis.extracted_features]
        analysis.verification_results = verification_results
        return analysis

    async def _verify_batch(
        self,
        hypotheses: List[Hypothesis],
        image_path: str
    ) -> List[VerificationResult]:
        tasks = [self._verify_single(h, image_path) for h in hypotheses]
        return await asyncio.gather(*tasks)

    async def _verify_single(
        self,
        hypothesis: Hypothesis,
        image_path: str
    ) -> VerificationResult:
        """
        Single hypothesis call, requesting JSON answer; thoughts optionally captured.
        """
        try:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()

            user_prompt = SINGLE_HYPOTHESIS_VERIFICATION_USER_PROMPT.format(
                image_path=image_path,
                hypothesis_json=json.dumps(hypothesis.model_dump(), ensure_ascii=False, indent=2)
            )

            # Build config with thinking if enabled
            gen_config = types.GenerateContentConfig()
            if self.include_thoughts:
                gen_config.thinking_config = types.ThinkingConfig(include_thoughts=True)
            # Enforce structured JSON in answer parts; model still may add plain text first, but this nudges compliance.
            gen_config.response_mime_type = "application/json"

            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=[
                    self.system_prompt,
                    user_prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
                ],
                config=gen_config
            )

            thought_segments, answer_text = self._separate_thought_and_answer(response)
            return self._parse_single_hypothesis_response(answer_text, hypothesis, thought_segments)

        except Exception as e:
            return self._fallback_error_result(hypothesis, f"Verification failed: {e}")

    # -------- Thought / Answer Separation -------- #

    def _separate_thought_and_answer(self, response) -> (List[str], str):
        """
        Extract thought parts (if any) vs answer parts.
        We only parse JSON from answer parts.
        """
        thought_segments: List[str] = []
        answer_segments: List[str] = []

        # Some SDK variants: response.candidates[0].content.parts
        candidates = getattr(response, "candidates", []) or []
        if not candidates:
            return thought_segments, getattr(response, "text", "")

        parts = getattr(candidates[0].content, "parts", []) or []
        for p in parts:
            txt = getattr(p, "text", None)
            if not txt:
                continue
            if getattr(p, "thought", False):
                # Internal reasoning
                thought_segments.append(txt.strip())
            else:
                answer_segments.append(txt)

        answer_text = "\n".join(answer_segments).strip()
        # Fallback if empty answer but we do have thoughts (rare misuse)
        if not answer_text and thought_segments:
            # We DO NOT parse JSON from raw thoughts; but we keep them in case we can salvage reasoning.
            pass
        return thought_segments, answer_text

    # -------- Parsing & Post-processing (extended with thought synthesis) -------- #

    def _parse_single_hypothesis_response(
        self,
        answer_text: str,
        original_hypothesis: Hypothesis,
        thought_segments: List[str]
    ) -> VerificationResult:
        json_obj = self._extract_best_json(answer_text)
        if json_obj is None:
            # Attempt salvage: if no JSON but we have thought segments, produce inconclusive fallback using synthesized reasoning
            reasoning = self._summarize_thoughts(thought_segments, original_hypothesis.statement) if thought_segments else "No valid JSON and no thoughts to summarize."
            return VerificationResult(
                hypothesis=original_hypothesis,
                reasoning=reasoning,
                evidence=[],
                conclusion="inconclusive: structured output missing"
            )

        hyp_block = json_obj.get("hypothesis", {})
        updated_hypothesis = Hypothesis(
            id=original_hypothesis.id,
            statement=original_hypothesis.statement,
            status=self._normalize_status(hyp_block.get("status", original_hypothesis.status)),
            confidence=self._clamp_float(hyp_block.get("confidence", original_hypothesis.confidence), 0.0, 1.0)
        )

        # Evidence
        evidence_objs: List[Evidence] = []
        for ev in json_obj.get("evidence", []):
            try:
                evidence_objs.append(
                    Evidence(
                        content=str(ev.get("content", "")).strip(),
                        type=EvidenceType(ev.get("type", "observation")),
                        supports_hypothesis=bool(ev.get("supports_hypothesis", True)),
                        confidence=self._clamp_float(ev.get("confidence", 0.5), 0.0, 1.0)
                    )
                )
            except Exception:
                continue

        reasoning = str(json_obj.get("reasoning", "")).strip()
        conclusion = str(json_obj.get("conclusion", "")).strip()

        updated_hypothesis = self._recompute_status_confidence(updated_hypothesis, evidence_objs, reasoning)

        # If reasoning weak or missing, enrich from thoughts
        if (not reasoning) or len(reasoning) < 25:
            if thought_segments:
                synth = self._summarize_thoughts(thought_segments, updated_hypothesis.statement)
                reasoning = (reasoning + "\n" + synth).strip() if reasoning else synth
            else:
                reasoning = self._synthesize_reasoning(updated_hypothesis, evidence_objs)

        # Ensure evidence references
        reasoning = self._ensure_evidence_references(reasoning, evidence_objs)

        if not conclusion:
            conclusion = self._synthesize_conclusion(updated_hypothesis, evidence_objs)

        return VerificationResult(
            hypothesis=updated_hypothesis,
            reasoning=reasoning,
            evidence=evidence_objs,
            conclusion=conclusion
        )

    # -------- Thought Summarization -------- #

    def _summarize_thoughts(self, thoughts: List[str], statement: str) -> str:
        """
        Compress raw thought segments into a sanitized, user-safe reasoning summary.
        Avoid exposing verbose internal chains; produce structured bullet reasoning.
        """
        if not thoughts:
            return "No internal thoughts available for synthesis."
        # Basic condensation: take last 2–3 segments (often more refined)
        trimmed = thoughts[-3:]
        joined = " ".join(seg.strip() for seg in trimmed if seg.strip())
        # Light normalization
        joined = re.sub(r"\s+", " ", joined)
        # Heuristic segmentation
        bullets = []
        # Simple lexical splits
        for chunk in re.split(r"(?<=[.;])\s+", joined):
            c = chunk.strip()
            if len(c) > 12 and not c.lower().startswith("final answer"):
                bullets.append(c)
        if not bullets:
            bullets = [joined]

        bullet_lines = "\n".join(f"- {b}" for b in bullets[:6])
        return (
            f"Synthesized internal analysis for hypothesis '{statement}':\n"
            f"{bullet_lines}\n"
            "End of synthesized internal reasoning."
        )

    # -------- Existing utilities (mostly unchanged except made internal) -------- #

    def _fallback_error_result(self, hypothesis: Hypothesis, msg: str) -> VerificationResult:
        return VerificationResult(
            hypothesis=hypothesis,
            reasoning=f"Process error: {msg}",
            evidence=[],
            conclusion="inconclusive: processing error prevented verification."
        )

    def _extract_best_json(self, text: str) -> Optional[dict]:
        text = text.strip()
        # Fenced code
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        candidates = []
        for block in fenced:
            c = self._try_load_json(block)
            if c:
                candidates.append(c)
        if not candidates:
            # Braces fallback
            brace_matches = re.findall(r"(\{[\s\S]*\})", text)
            for bm in brace_matches:
                c = self._try_load_json(bm)
                if c:
                    candidates.append(c)
        if not candidates:
            # Attempt raw parse
            c = self._try_load_json(text)
            if c:
                candidates.append(c)
        if not candidates:
            return None
        for c in candidates:
            if "hypothesis" in c:
                return c
        return candidates[0]

    @staticmethod
    def _try_load_json(s: str) -> Optional[dict]:
        try:
            return json.loads(s)
        except Exception:
            return None

    @staticmethod
    def _normalize_status(status_val: Any) -> HypothesisStatus:
        try:
            sv = str(status_val).strip().lower()
            if sv in {"verified", "rejected", "inconclusive", "pending"}:
                return HypothesisStatus(sv)
        except Exception:
            pass
        return HypothesisStatus.INCONCLUSIVE

    @staticmethod
    def _clamp_float(v: Any, lo: float, hi: float) -> float:
        try:
            f = float(v)
        except Exception:
            f = lo
        return max(lo, min(hi, f))

    def _recompute_status_confidence(
        self,
        hypothesis: Hypothesis,
        evidence: List[Evidence],
        reasoning: str
    ) -> Hypothesis:
        if not evidence:
            if hypothesis.status == HypothesisStatus.VERIFIED:
                hypothesis.status = HypothesisStatus.INCONCLUSIVE
            if hypothesis.status == HypothesisStatus.INCONCLUSIVE and hypothesis.confidence > 0.65:
                hypothesis.confidence = 0.55
            if hypothesis.status == HypothesisStatus.PENDING:
                hypothesis.confidence = min(hypothesis.confidence, 0.55)
            return hypothesis

        support = sum(ev.confidence for ev in evidence if ev.supports_hypothesis)
        oppose = sum(ev.confidence for ev in evidence if not ev.supports_hypothesis)

        if support > 0.9 and oppose < 0.25:
            derived = HypothesisStatus.VERIFIED
        elif oppose > support * 1.2 and oppose > 0.6:
            derived = HypothesisStatus.REJECTED
        elif (support < 0.5 and oppose < 0.5) or abs(support - oppose) < 0.25:
            derived = HypothesisStatus.INCONCLUSIVE
        else:
            if support > oppose:
                derived = HypothesisStatus.VERIFIED if support >= 1.2 * oppose and support > 0.8 else HypothesisStatus.INCONCLUSIVE
            else:
                derived = HypothesisStatus.REJECTED if oppose >= 1.2 * support and oppose > 0.8 else HypothesisStatus.INCONCLUSIVE

        hypothesis.status = derived
        if derived == HypothesisStatus.VERIFIED:
            hypothesis.confidence = self._clamp_float(0.70 + min(0.25, support / 4.0), 0.70, 0.95)
        elif derived == HypothesisStatus.REJECTED:
            hypothesis.confidence = self._clamp_float(0.05 + min(0.20, oppose / 3.0), 0.05, 0.30)
        elif derived == HypothesisStatus.INCONCLUSIVE:
            total = support + oppose + 1e-6
            leaning = abs(support - oppose) / total
            base = 0.30 + 0.35 * (1 - leaning)
            hypothesis.confidence = self._clamp_float(base, 0.30, 0.65)
        else:
            hypothesis.confidence = self._clamp_float(hypothesis.confidence, 0.40, 0.55)
        return hypothesis

    def _synthesize_reasoning(self, hypothesis: Hypothesis, evidence: List[Evidence]) -> str:
        if not evidence:
            return (
                f"No evidence extracted; hypothesis remains {hypothesis.status.value} due to insufficient data."
            )
        lines = [f"Evaluating hypothesis '{hypothesis.statement}':"]
        for idx, ev in enumerate(evidence, start=1):
            stance = "supports" if ev.supports_hypothesis else "contradicts"
            lines.append(
                f"Evidence #{idx} ({ev.type.value}, conf {ev.confidence:.2f}) {stance}: {ev.content}"
            )
        lines.append(
            f"Result => status={hypothesis.status.value}, confidence={hypothesis.confidence:.2f} after weighting evidence."
        )
        return "\n".join(lines)

    def _synthesize_conclusion(self, hypothesis: Hypothesis, evidence: List[Evidence]) -> str:
        if hypothesis.status == HypothesisStatus.VERIFIED:
            reason = "consistent supporting evidence"
        elif hypothesis.status == HypothesisStatus.REJECTED:
            reason = "contradictory evidence predominates"
        elif hypothesis.status == HypothesisStatus.INCONCLUSIVE:
            reason = "mixed or insufficient evidence"
        else:
            reason = "verification not currently possible"
        return f"{hypothesis.status.value}: {reason}"

    def _ensure_evidence_references(self, reasoning: str, evidence: List[Evidence]) -> str:
        if not evidence:
            return reasoning
        if re.search(r"Evidence\s+#\d+", reasoning, flags=re.IGNORECASE):
            return reasoning
        appendix_lines = ["", "Evidence Index:"]
        for i, ev in enumerate(evidence, start=1):
            appendix_lines.append(f"Evidence #{i}: {ev.content[:120]}")
        return reasoning.strip() + "\n" + "\n".join(appendix_lines)