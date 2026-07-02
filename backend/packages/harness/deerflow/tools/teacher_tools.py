import asyncio
import json
import logging
import os
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, ValidationError

from deerflow.config import get_app_config
from deerflow.models import create_chat_model
from deerflow.teacher_knowledge import KNOWLEDGE_DETAIL_MAP, MATH_KNOWLEDGE_TYPES
from deerflow.teacher_metadata import ABILITY_TAGS, ERROR_TAGS, METHOD_TAGS, normalize_ability_tags, normalize_error_tags, normalize_method_tags, normalize_stage
from deerflow.teacher_persistence import normalize_difficulty, normalize_knowledge_pair, normalize_knowledge_tags, normalize_problem_type, normalize_subject, persist_safely_async, retrieve_similar_problems
from deerflow.teacher_profile import (
    parse_student_profile_markdown,
    read_student_profile_summary,
    render_student_profile_markdown,
    update_student_profile_manual,
    write_student_profile_summary,
)

_DEFAULT_TEXT_MODEL_ENV = "DEER_FLOW_TEACHER_MODEL"
_SOLVE_MODEL_ENV = "DEER_FLOW_TEACHER_SOLVE_MODEL"
_RECOMMEND_MODEL_ENV = "DEER_FLOW_TEACHER_RECOMMEND_MODEL"
_OCR_MODEL_ENV = "DEER_FLOW_TEACHER_OCR_MODEL"
_EVALUATE_EXPLANATION_MODEL_ENV = "DEER_FLOW_TEACHER_EVALUATE_EXPLANATION_MODEL"

logger = logging.getLogger(__name__)


class SolveCoreResult(BaseModel):
    answer: str = ""
    steps: list[str] = []
    explanation: str = ""


class SolveKnowledgesResult(BaseModel):
    knowledges: list[str] = []


class SolveErrorAnalysisResult(BaseModel):
    error_analysis: str | None = None


class SolveWeakPointsResult(BaseModel):
    weak_knowledge_candidates: list[str] = []
    weak_ability_candidates: list[str] = []


class SolveClassificationResult(BaseModel):
    problem_type: str | None = None
    difficulty: str | None = None
    knowledge_type: str | None = None
    knowledge_detail: str | None = None
    stage: str | None = None
    ability_tags: list[str] = []
    method_tags: list[str] = []
    error_tags: list[str] = []


class SolveSubjectResult(BaseModel):
    subject: str | None = None


class RecommendedProblem(BaseModel):
    title: str
    question: str = ""
    practice_objective: str = ""
    similarity: str = ""


class RecommendSimilarProblemsResult(BaseModel):
    items: list[RecommendedProblem] = []
    message: str = ""


class OcrProblemImageResult(BaseModel):
    text: str = ""
    message: str = ""


class EvaluateStudentExplanationResult(BaseModel):
    understood: list[str] = []
    misconception: str = ""
    gap_type: str = ""
    remediation: str = ""
    followup_question: str = ""
    should_update_profile: bool = False
    weak_knowledge_candidates: list[str] = []
    weak_ability_candidates: list[str] = []


def _normalize_knowledges(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    return [text] if text else []


def _get_teacher_model_name(env_name: str) -> str | None:
    return os.getenv(env_name) or os.getenv(_DEFAULT_TEXT_MODEL_ENV)


def _model_supports_vision(model_name: str | None) -> bool:
    app_config = get_app_config()
    if model_name is None:
        model_name = app_config.models[0].name if app_config.models else None
    if model_name is None:
        return False
    model_config = app_config.get_model_config(model_name)
    return bool(model_config and model_config.supports_vision)


def _extract_response_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].startswith("```"):
            candidates.append("\n".join(lines[1:-1]).strip())
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                continue
            try:
                data = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict):
            return data
    return None


def _build_student_profile_context(student_id: str | None) -> str:
    if not student_id:
        return ""
    return f"student_profile_l0_is_injected_in_system_prompt: true\nstudent_id: {student_id}"


async def _invoke_json_tool(model_name: str | None, system_instruction: str, user_prompt: str) -> tuple[dict[str, Any] | None, str]:
    model = create_chat_model(name=model_name, thinking_enabled=False)
    response = await model.ainvoke(
        [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]
    )
    raw_text = _extract_response_text(getattr(response, "content", ""))
    return _extract_json_object(raw_text), raw_text


def _solve_core_system_instruction() -> str:
    return (
        "You are a math and science tutoring tool. "
        "Solve the problem and return strict JSON only with keys: answer, steps, explanation. "
        "steps must be an array of strings. "
        "Do not include markdown fences or extra commentary outside the JSON object."
    )


def _solve_knowledges_system_instruction() -> str:
    return (
        "You extract concise knowledge tags for a solved educational problem. "
        "Return strict JSON only with key: knowledges. "
        "knowledges must be an array of short strings."
    )


def _solve_error_analysis_system_instruction() -> str:
    return (
        "You analyze the likely mistake a student may make on this problem. "
        "Return strict JSON only with key: error_analysis. "
        "If there is no clear mistake pattern from the problem context, set error_analysis to null."
    )


def _solve_subject_system_instruction() -> str:
    return (
        "You classify the subject of an educational problem. "
        "Return strict JSON only with key: subject. "
        "subject must be Chinese only and must be one of: 数学, 语文, 英语, 物理, 化学, 生物."
    )


def _solve_weak_points_system_instruction() -> str:
    return (
        "You infer likely weak knowledge points and weak abilities from a student's problem-solving context. "
        "Return strict JSON only with keys: weak_knowledge_candidates, weak_ability_candidates. "
        "Both keys must be arrays of strings."
    )


def _format_knowledge_taxonomy_for_prompt() -> str:
    lines: list[str] = []
    for knowledge_type in MATH_KNOWLEDGE_TYPES:
        details = "；".join(KNOWLEDGE_DETAIL_MAP.get(knowledge_type, ()))
        lines.append(f"- {knowledge_type}: {details}")
    return "\n".join(lines)


def _solve_classification_system_instruction() -> str:
    return (
        "You classify an educational problem into problem metadata for a structured teaching problem bank. "
        "Return strict JSON only with keys: problem_type, difficulty, knowledge_type, knowledge_detail, stage, ability_tags, method_tags, error_tags. "
        "problem_type must be Chinese only and must be one of: 单选, 多选, 填空, 大题. "
        "difficulty must be Chinese only and must be one of: 简单, 中等, 困难. "
        "stage must be Chinese only and must be one of: 小学, 初中, 高中. "
        "ability_tags, method_tags, and error_tags must be arrays chosen only from the fixed tag sets below. "
        "knowledge_type and knowledge_detail must be Chinese only, never English, never pinyin. "
        "You must choose knowledge_type and knowledge_detail from the following fixed Chinese taxonomy. "
        "If no exact item fits, choose the closest Chinese item from the taxonomy instead of inventing a new term.\n\n"
        f"Ability tags: {'、'.join(ABILITY_TAGS)}\n"
        f"Method tags: {'、'.join(METHOD_TAGS)}\n"
        f"Error tags: {'、'.join(ERROR_TAGS)}\n\n"
        f"Knowledge taxonomy:\n{_format_knowledge_taxonomy_for_prompt()}"
    )


def _build_solve_context_prompt(question: str, student_id: str | None, image_url: str | None, subject: str | None, grade: str | None) -> str:
    profile_context = _build_student_profile_context(student_id)
    return (
        f"student_id: {student_id or 'unknown'}\n"
        f"subject: {subject or ''}\n"
        f"grade: {grade or ''}\n"
        f"image_url: {image_url or ''}\n"
        f"question:\n{question}\n\n"
        f"student_profile_markdown:\n{profile_context}\n"
    )


def _build_solve_core_prompt(question: str, student_id: str | None, image_url: str | None, subject: str | None, grade: str | None) -> str:
    return _build_solve_context_prompt(question, student_id, image_url, subject, grade)


def _build_solve_knowledges_prompt(
    question: str,
    answer: str,
    steps: list[str],
    explanation: str,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> str:
    return (
        _build_solve_context_prompt(question, student_id, image_url, subject, grade)
        + "\ncore_solution:\n"
        + json.dumps({"answer": answer, "steps": steps, "explanation": explanation}, ensure_ascii=False)
    )


def _build_solve_error_analysis_prompt(
    question: str,
    answer: str,
    steps: list[str],
    explanation: str,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> str:
    return (
        _build_solve_context_prompt(question, student_id, image_url, subject, grade)
        + "\ncore_solution:\n"
        + json.dumps({"answer": answer, "steps": steps, "explanation": explanation}, ensure_ascii=False)
    )


def _build_solve_subject_prompt(
    question: str,
    answer: str,
    steps: list[str],
    explanation: str,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> str:
    return (
        _build_solve_context_prompt(question, student_id, image_url, subject, grade)
        + "\ncore_solution:\n"
        + json.dumps({"answer": answer, "steps": steps, "explanation": explanation}, ensure_ascii=False)
    )


def _build_solve_weak_points_prompt(
    question: str,
    answer: str,
    steps: list[str],
    explanation: str,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> str:
    return (
        _build_solve_context_prompt(question, student_id, image_url, subject, grade)
        + "\ncore_solution:\n"
        + json.dumps(
            {
                "answer": answer,
                "steps": steps,
                "explanation": explanation,
            },
            ensure_ascii=False,
        )
    )


def _build_solve_classification_prompt(
    question: str,
    answer: str,
    steps: list[str],
    explanation: str,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> str:
    return (
        _build_solve_context_prompt(question, student_id, image_url, subject, grade)
        + "\ncore_solution:\n"
        + json.dumps(
            {
                "answer": answer,
                "steps": steps,
                "explanation": explanation,
            },
            ensure_ascii=False,
        )
    )


def _recommend_system_instruction() -> str:
    return (
        "You recommend a small set of similar practice problems for a student. "
        "Return strict JSON only with keys: items, message. "
        "items must be an array of objects with keys: title, question, practice_objective, similarity."
    )


def _build_recommendation_item(problem: dict[str, Any]) -> dict[str, str]:
    title = str(problem.get("problem_type") or problem.get("knowledge_type") or f"题目 {problem.get('qid')}").strip()
    knowledge_detail = str(problem.get("knowledge_detail") or "").strip()
    difficulty = str(problem.get("difficulty") or "").strip()
    practice_bits = [bit for bit in [knowledge_detail, f"难度 {difficulty}" if difficulty else ""] if bit]
    similarity_bits = [bit for bit in [problem.get("knowledge_type"), knowledge_detail, difficulty] if bit]
    return {
        "title": title,
        "question": str(problem.get("question") or ""),
        "practice_objective": "巩固" + "，".join(practice_bits) if practice_bits else "巩固同类题型方法",
        "similarity": str(problem.get("recommend_reason") or ("同类练习：" + " / ".join(str(bit) for bit in similarity_bits) if similarity_bits else "同类知识点与题型")),
    }


def _build_recommend_prompt(question: str, student_id: str | None, knowledges: list[str] | None) -> str:
    profile_context = _build_student_profile_context(student_id)
    return (
        f"student_id: {student_id or 'unknown'}\n"
        f"question:\n{question}\n\n"
        f"knowledges: {json.dumps(knowledges or [], ensure_ascii=False)}\n\n"
        f"student_profile_markdown:\n{profile_context}\n"
    )


def _ocr_system_instruction() -> str:
    return (
        "You are an OCR-style extraction tool for educational problem images. "
        "Return strict JSON only with keys: text, message. "
        "Extract the problem text cleanly. If the image cannot be read, explain briefly in message."
    )


def _evaluate_student_explanation_system_instruction() -> str:
    return (
        "You evaluate whether a student's explanation shows real understanding. "
        "Return strict JSON only with keys: understood, misconception, gap_type, remediation, followup_question, should_update_profile, weak_knowledge_candidates, weak_ability_candidates. "
        "understood, weak_knowledge_candidates, and weak_ability_candidates must be arrays of short strings. "
        "gap_type must be one of: concept_gap, procedure_gap, transfer_gap, carelessness_or_expression_gap, none."
    )


def _build_evaluate_student_explanation_prompt(
    question: str,
    student_explanation: str,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
    reference_answer: str | None,
    reference_steps: list[str] | None,
    reference_explanation: str | None,
    reference_knowledges: list[str] | None,
) -> str:
    return (
        _build_solve_context_prompt(question, student_id, image_url, subject, grade)
        + "\nstudent_explanation:\n"
        + (student_explanation or "")
        + "\n\nreference_solution:\n"
        + json.dumps(
            {
                "answer": reference_answer or "",
                "steps": reference_steps or [],
                "explanation": reference_explanation or "",
                "knowledges": reference_knowledges or [],
            },
            ensure_ascii=False,
        )
    )


async def _invoke_ocr_model(model_name: str | None, image_url: str) -> tuple[dict[str, Any] | None, str]:
    model = create_chat_model(name=model_name, thinking_enabled=False)
    response = await model.ainvoke(
        [
            {
                "role": "system",
                "content": _ocr_system_instruction(),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the educational problem text from this image and return strict JSON."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
    )
    raw_text = _extract_response_text(getattr(response, "content", ""))
    return _extract_json_object(raw_text), raw_text


async def _run_solve_core(
    model_name: str | None,
    question: str,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> tuple[SolveCoreResult, Any]:
    payload_data, raw_text = await _invoke_json_tool(
        model_name,
        _solve_core_system_instruction(),
        _build_solve_core_prompt(question, student_id, image_url, subject, grade),
    )
    if payload_data is None:
        raise ValueError("model did not return valid JSON")
    return SolveCoreResult.model_validate(payload_data), payload_data or raw_text


async def _run_solve_knowledges(
    model_name: str | None,
    question: str,
    core: SolveCoreResult,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> tuple[list[str], Any]:
    payload_data, raw_text = await _invoke_json_tool(
        model_name,
        _solve_knowledges_system_instruction(),
        _build_solve_knowledges_prompt(question, core.answer, core.steps, core.explanation, student_id, image_url, subject, grade),
    )
    if payload_data is None:
        raise ValueError("model did not return valid JSON")
    parsed = SolveKnowledgesResult.model_validate(payload_data)
    return _normalize_knowledges(parsed.knowledges), payload_data or raw_text


async def _run_solve_error_analysis(
    model_name: str | None,
    question: str,
    core: SolveCoreResult,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> tuple[str | None, Any]:
    payload_data, raw_text = await _invoke_json_tool(
        model_name,
        _solve_error_analysis_system_instruction(),
        _build_solve_error_analysis_prompt(question, core.answer, core.steps, core.explanation, student_id, image_url, subject, grade),
    )
    if payload_data is None:
        raise ValueError("model did not return valid JSON")
    parsed = SolveErrorAnalysisResult.model_validate(payload_data)
    return parsed.error_analysis, payload_data or raw_text


async def _run_solve_subject(
    model_name: str | None,
    question: str,
    core: SolveCoreResult,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> tuple[str | None, Any]:
    payload_data, raw_text = await _invoke_json_tool(
        model_name,
        _solve_subject_system_instruction(),
        _build_solve_subject_prompt(question, core.answer, core.steps, core.explanation, student_id, image_url, subject, grade),
    )
    if payload_data is None:
        raise ValueError("model did not return valid JSON")
    parsed = SolveSubjectResult.model_validate(payload_data)
    return normalize_subject(parsed.subject), payload_data or raw_text


async def _run_solve_weak_points(
    model_name: str | None,
    question: str,
    core: SolveCoreResult,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> tuple[SolveWeakPointsResult, Any]:
    payload_data, raw_text = await _invoke_json_tool(
        model_name,
        _solve_weak_points_system_instruction(),
        _build_solve_weak_points_prompt(
            question,
            core.answer,
            core.steps,
            core.explanation,
            student_id,
            image_url,
            subject,
            grade,
        ),
    )
    if payload_data is None:
        raise ValueError("model did not return valid JSON")
    return SolveWeakPointsResult.model_validate(payload_data), payload_data or raw_text


async def _run_solve_classification(
    model_name: str | None,
    question: str,
    core: SolveCoreResult,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
) -> tuple[SolveClassificationResult, Any]:
    payload_data, raw_text = await _invoke_json_tool(
        model_name,
        _solve_classification_system_instruction(),
        _build_solve_classification_prompt(
            question,
            core.answer,
            core.steps,
            core.explanation,
            student_id,
            image_url,
            subject,
            grade,
        ),
    )
    if payload_data is None:
        raise ValueError("model did not return valid JSON")
    return SolveClassificationResult.model_validate(payload_data), payload_data or raw_text


@tool("solve_problem", parse_docstring=True)
async def solve_problem_tool(
    question: str,
    student_id: str | None = None,
    image_url: str | None = None,
    subject: str | None = None,
    grade: str | None = None,
) -> dict[str, Any]:
    """Solve a student problem via DeerFlow's configured chat model.

    Args:
        question: The problem text. For image-based questions, pass the user question or OCR text.
        student_id: Optional student identifier for personalization and tracing.
        image_url: Optional uploaded image URL or externally accessible file URL for image-based questions.
        subject: Optional subject hint.
        grade: Optional grade hint.
    """
    subject = normalize_subject(subject)
    model_name = _get_teacher_model_name(_SOLVE_MODEL_ENV)
    try:
        core, core_raw = await _run_solve_core(model_name, question, student_id, image_url, subject, grade)
    except (ValidationError, ValueError) as exc:
        return {
            "status": "error",
            "answer": "",
            "steps": [],
            "explanation": f"Failed to parse solver model output: {exc}",
            "knowledges": [],
            "error_analysis": None,
            "weak_knowledge_candidates": [],
            "weak_ability_candidates": [],
        }
    except Exception as exc:
        return {
            "status": "error",
            "answer": "",
            "steps": [],
            "explanation": f"Failed to call DeerFlow model for solve_problem: {exc}",
            "knowledges": [],
            "error_analysis": None,
            "weak_knowledge_candidates": [],
            "weak_ability_candidates": [],
        }

    knowledges_task = _run_solve_knowledges(model_name, question, core, student_id, image_url, subject, grade)
    error_analysis_task = _run_solve_error_analysis(model_name, question, core, student_id, image_url, subject, grade)
    subject_task = _run_solve_subject(model_name, question, core, student_id, image_url, subject, grade)
    weak_points_task = _run_solve_weak_points(model_name, question, core, student_id, image_url, subject, grade)
    classification_task = _run_solve_classification(model_name, question, core, student_id, image_url, subject, grade)

    knowledges_result, error_analysis_result, subject_result, weak_points_result, classification_result = await asyncio.gather(
        knowledges_task,
        error_analysis_task,
        subject_task,
        weak_points_task,
        classification_task,
        return_exceptions=True,
    )

    raw_knowledges: list[str] = []
    knowledges: list[str] = []
    knowledges_raw: Any = None
    if not isinstance(knowledges_result, Exception):
        raw_knowledges, knowledges_raw = knowledges_result

    error_analysis: str | None = None
    error_analysis_raw: Any = None
    if not isinstance(error_analysis_result, Exception):
        error_analysis, error_analysis_raw = error_analysis_result

    subject_raw: Any = None
    if not isinstance(subject_result, Exception):
        subject, subject_raw = subject_result

    weak_knowledge_candidates: list[str] = []
    weak_ability_candidates: list[str] = []
    weak_points_raw: Any = None
    if not isinstance(weak_points_result, Exception):
        weak_points, weak_points_raw = weak_points_result
        weak_knowledge_candidates = weak_points.weak_knowledge_candidates
        weak_ability_candidates = weak_points.weak_ability_candidates

    problem_type: str | None = None
    difficulty: str | None = None
    knowledge_type: str | None = None
    knowledge_detail: str | None = None
    stage: str | None = None
    ability_tags: list[str] = []
    method_tags: list[str] = []
    error_tags: list[str] = []
    classification_raw: Any = None
    if not isinstance(classification_result, Exception):
        classification, classification_raw = classification_result
        problem_type = normalize_problem_type(classification.problem_type)
        difficulty = normalize_difficulty(classification.difficulty)
        stage = normalize_stage(classification.stage, grade)
        ability_tags = normalize_ability_tags(classification.ability_tags)
        method_tags = normalize_method_tags(classification.method_tags)
        error_tags = normalize_error_tags(classification.error_tags)
        knowledge_type, knowledge_detail, knowledges = normalize_knowledge_tags(classification.knowledge_type, classification.knowledge_detail, raw_knowledges)
    else:
        stage = normalize_stage(None, grade)
        knowledge_type, knowledge_detail, knowledges = normalize_knowledge_tags(None, None, raw_knowledges)

    result = {
        "status": "ok",
        "answer": core.answer,
        "steps": core.steps,
        "explanation": core.explanation,
        "knowledges": knowledges,
        "error_analysis": error_analysis,
        "problem_type": problem_type,
        "difficulty": difficulty,
        "knowledge_type": knowledge_type,
        "knowledge_detail": knowledge_detail,
        "stage": stage,
        "ability_tags": ability_tags,
        "method_tags": method_tags,
        "error_tags": error_tags,
        "weak_knowledge_candidates": weak_knowledge_candidates,
        "weak_ability_candidates": weak_ability_candidates,
        "raw": {
            "core": core_raw,
            "knowledges": knowledges_raw,
            "error_analysis": error_analysis_raw,
            "subject": subject_raw,
            "weak_points": weak_points_raw,
            "classification": classification_raw,
        },
    }
    logger.info(
        "solve_problem generated result: student_id=%r subject=%r grade=%r answer=%r steps=%s knowledges=%s error_analysis=%r weak_knowledge_candidates=%s weak_ability_candidates=%s",
        student_id,
        subject,
        grade,
        result["answer"][:200],
        json.dumps(result["steps"], ensure_ascii=False),
        json.dumps(result["knowledges"], ensure_ascii=False),
        result["error_analysis"][:200] if isinstance(result["error_analysis"], str) else result["error_analysis"],
        json.dumps(result["weak_knowledge_candidates"], ensure_ascii=False),
        json.dumps(result["weak_ability_candidates"], ensure_ascii=False),
    )
    persistence, persistence_error = await persist_safely_async(
        question=question,
        student_id=student_id,
        image_url=image_url,
        subject=subject,
        grade=grade,
        result=result,
    )
    if persistence is not None:
        result["persistence"] = persistence
        logger.info(
            "solve_problem persistence succeeded: student_id=%r problem_id=%s problem_detail_id=%s student_profile_path=%r",
            student_id,
            persistence.get("problem_id"),
            persistence.get("problem_detail_id"),
            persistence.get("student_profile_path"),
        )
    elif persistence_error:
        result["persistence_error"] = persistence_error
        logger.warning("solve_problem persistence failed: student_id=%r error=%s", student_id, persistence_error)
    return result


@tool("read_student_profile", parse_docstring=True)
def read_student_profile_tool(student_id: str) -> str:
    """Read the markdown summary for a student profile.

    Args:
        student_id: Student identifier provided by the caller.
    """
    return read_student_profile_summary(student_id)


@tool("update_student_profile", parse_docstring=True)
def update_student_profile_tool(
    student_id: str,
    student_name: str | None = None,
    grade: str | None = None,
    subject: str | None = None,
    weak_knowledge: list[str] | None = None,
    weak_ability: list[str] | None = None,
    preferences: list[str] | None = None,
    recent_summary: str | None = None,
) -> dict[str, str]:
    """Update the local markdown summary for a student profile.

    Args:
        student_id: Student identifier provided by the caller.
        student_name: Optional student name for the profile.
        grade: Optional student grade for the profile.
        subject: Optional primary subject for the profile.
        weak_knowledge: Weak knowledge points to summarize into the profile.
        weak_ability: Weak ability points to summarize into the profile.
        preferences: Learning preferences or tutoring preferences.
        recent_summary: Short summary of the latest learning observations.
    """
    path = update_student_profile_manual(
        student_id,
        student_name=student_name,
        grade=grade,
        subject=subject,
        weak_knowledge=weak_knowledge,
        weak_ability=weak_ability,
        preferences=preferences,
        recent_summary=recent_summary,
    )
    return {"status": "ok", "path": str(path)}


@tool("sync_student_profile", parse_docstring=True)
async def sync_student_profile_tool(student_id: str) -> dict[str, Any]:
    """Validate and reuse the local student profile markdown summary.

    Args:
        student_id: Student identifier for the local mirrored markdown profile.
    """
    markdown = read_student_profile_summary(student_id)
    if not markdown:
        return {"status": "empty", "message": "No local student profile markdown exists yet."}
    profile = parse_student_profile_markdown(student_id, markdown)
    path = write_student_profile_summary(student_id, render_student_profile_markdown(profile))
    return {"status": "ok", "path": str(path)}


@tool("recommend_similar_problems", parse_docstring=True)
async def recommend_similar_problems_tool(
    question: str,
    student_id: str | None = None,
    knowledges: list[str] | None = None,
    subject: str | None = None,
    difficulty: str | None = None,
    knowledge_type: str | None = None,
    knowledge_detail: str | None = None,
    grade: str | None = None,
    stage: str | None = None,
    ability_tags: list[str] | None = None,
    method_tags: list[str] | None = None,
    error_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Recommend similar problems via the structured problem bank first, then DeerFlow's configured chat model.

    Args:
        question: The current problem or topic text.
        student_id: Optional student identifier for personalized retrieval.
        knowledges: Optional knowledge tags to bias retrieval.
        subject: Optional subject hint for bank retrieval.
        difficulty: Optional difficulty hint for bank retrieval.
        knowledge_type: Optional coarse-grained knowledge category.
        knowledge_detail: Optional fine-grained knowledge detail.
        grade: Optional grade hint for global bank retrieval.
        stage: Optional stage hint, one of 小学/初中/高中.
        ability_tags: Optional controlled ability tags.
        method_tags: Optional controlled solution-method tags.
        error_tags: Optional controlled common-error tags.
    """
    try:
        bank_items = await asyncio.to_thread(
            retrieve_similar_problems,
            question=question,
            student_id=student_id,
            subject=subject,
            knowledges=knowledges,
            difficulty=difficulty,
            knowledge_type=knowledge_type,
            knowledge_detail=knowledge_detail,
            grade=grade,
            stage=stage,
            ability_tags=ability_tags,
            method_tags=method_tags,
            error_tags=error_tags,
            limit=3,
        )
    except Exception as exc:
        logger.warning("recommend_similar_problems bank retrieval failed: %s", exc)
        bank_items = []

    if bank_items:
        return {
            "status": "ok",
            "items": [_build_recommendation_item(item) for item in bank_items],
            "message": "已优先从题库中挑选同类题目。",
            "raw": {"source": "problem_bank", "items": bank_items},
        }

    model_name = _get_teacher_model_name(_RECOMMEND_MODEL_ENV)
    try:
        payload_data, raw_text = await _invoke_json_tool(
            model_name,
            _recommend_system_instruction(),
            _build_recommend_prompt(question, student_id, knowledges),
        )
        if payload_data is None:
            raise ValueError("model did not return valid JSON")
        parsed = RecommendSimilarProblemsResult.model_validate(payload_data)
    except (ValidationError, ValueError) as exc:
        return {
            "status": "error",
            "items": [],
            "message": f"Failed to parse recommender model output: {exc}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "items": [],
            "message": f"Failed to call DeerFlow model for recommend_similar_problems: {exc}",
        }

    return {
        "status": "ok",
        "items": [item.model_dump() for item in parsed.items],
        "message": parsed.message,
        "raw": payload_data or raw_text,
    }


@tool("ocr_problem_image", parse_docstring=True)
async def ocr_problem_image_tool(image_url: str) -> dict[str, Any]:
    """Extract problem text from an image URL using a DeerFlow-configured vision model.

    Args:
        image_url: An externally accessible image URL.
    """
    model_name = _get_teacher_model_name(_OCR_MODEL_ENV)
    if not _model_supports_vision(model_name):
        return {
            "status": "error",
            "text": "",
            "message": "ocr_problem_image requires a configured vision-capable DeerFlow model alias.",
        }

    try:
        payload_data, raw_text = await _invoke_ocr_model(model_name, image_url)
        if payload_data is None:
            raise ValueError("model did not return valid JSON")
        parsed = OcrProblemImageResult.model_validate(payload_data)
    except (ValidationError, ValueError) as exc:
        return {
            "status": "error",
            "text": "",
            "message": f"Failed to parse OCR model output: {exc}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "text": "",
            "message": f"Failed to call DeerFlow model for ocr_problem_image: {exc}",
        }

    return {
        "status": "ok",
        "text": parsed.text,
        "message": parsed.message,
        "raw": payload_data or raw_text,
    }


@tool("evaluate_student_explanation", parse_docstring=True)
async def evaluate_student_explanation_tool(
    question: str,
    student_explanation: str,
    student_id: str | None = None,
    wrong_input: str | None = None,
    image_url: str | None = None,
    subject: str | None = None,
    grade: str | None = None,
    reference_answer: str | None = None,
    reference_steps: list[str] | None = None,
    reference_explanation: str | None = None,
    reference_knowledges: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a student's explanation reflects real understanding.

    Args:
        question: The original problem or concept prompt.
        student_explanation: The student's own explanation in natural language.
        student_id: Optional student identifier for personalization and tracing.
        wrong_input: Optional wrong answer or wrong solution supplied earlier.
        image_url: Optional uploaded image URL for the original problem.
        subject: Optional subject hint.
        grade: Optional grade hint.
        reference_answer: Optional trusted final answer.
        reference_steps: Optional trusted steps for the reference solution.
        reference_explanation: Optional trusted explanation of the solution.
        reference_knowledges: Optional knowledge tags for the problem.
    """
    model_name = _get_teacher_model_name(_EVALUATE_EXPLANATION_MODEL_ENV)
    try:
        payload_data, raw_text = await _invoke_json_tool(
            model_name,
            _evaluate_student_explanation_system_instruction(),
            _build_evaluate_student_explanation_prompt(
                question,
                student_explanation,
                student_id,
                    image_url,
                subject,
                grade,
                reference_answer,
                reference_steps,
                reference_explanation,
                reference_knowledges,
            ),
        )
        if payload_data is None:
            raise ValueError("model did not return valid JSON")
        parsed = EvaluateStudentExplanationResult.model_validate(payload_data)
    except (ValidationError, ValueError) as exc:
        return {
            "status": "error",
            "understood": [],
            "misconception": "",
            "gap_type": "",
            "remediation": "",
            "followup_question": "",
            "should_update_profile": False,
            "weak_knowledge_candidates": [],
            "weak_ability_candidates": [],
            "message": f"Failed to parse explanation evaluator output: {exc}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "understood": [],
            "misconception": "",
            "gap_type": "",
            "remediation": "",
            "followup_question": "",
            "should_update_profile": False,
            "weak_knowledge_candidates": [],
            "weak_ability_candidates": [],
            "message": f"Failed to call DeerFlow model for evaluate_student_explanation: {exc}",
        }

    return {
        "status": "ok",
        "understood": parsed.understood,
        "misconception": parsed.misconception,
        "gap_type": parsed.gap_type,
        "remediation": parsed.remediation,
        "followup_question": parsed.followup_question,
        "should_update_profile": parsed.should_update_profile,
        "weak_knowledge_candidates": parsed.weak_knowledge_candidates,
        "weak_ability_candidates": parsed.weak_ability_candidates,
        "raw": payload_data or raw_text,
    }
