from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
import time
from typing import Any

import logging

from deerflow.teacher_knowledge import KNOWLEDGE_DETAIL_MAP, MATH_KNOWLEDGE_TYPES
from deerflow.teacher_metadata import normalize_ability_tags, normalize_error_tags, normalize_method_tags, normalize_stage
from deerflow.teacher_profile import parse_student_profile_markdown, read_student_profile_summary, update_student_profile_from_observation

_MYSQL_TABLE = os.getenv("DEER_FLOW_TEACHER_MYSQL_TABLE", "question_basic_info")
_MONGO_COLLECTION = os.getenv("DEER_FLOW_TEACHER_MONGO_COLLECTION", "teacher_problem_details")
_DEFAULT_ENV_VALUES = {
    "MYSQL_HOST": "127.0.0.1",
    "MYSQL_DATABASE": "education",
    "MYSQL_USER": "root",
    "MYSQL_PASSWORD": "123456",
    "MONGODB_URI": "mongodb://127.0.0.1:27017",
    "MONGODB_DATABASE": "education",
}
_DEFAULT_STUDENT_ID = "522025320226"
logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _mysql_enabled() -> bool:
    return True


def _mongo_enabled() -> bool:
    return True


def _mysql_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
        if name in _DEFAULT_ENV_VALUES:
            return _DEFAULT_ENV_VALUES[name]
    return default


def _mysql_connection():
    from pymysql import connect
    from pymysql.cursors import DictCursor

    return connect(
        host=_mysql_env("DEER_FLOW_TEACHER_MYSQL_HOST", "MYSQL_HOST"),
        port=int(_mysql_env("DEER_FLOW_TEACHER_MYSQL_PORT", "MYSQL_PORT", default="3306") or "3306"),
        user=_mysql_env("DEER_FLOW_TEACHER_MYSQL_USER", "MYSQL_USER"),
        password=_mysql_env("DEER_FLOW_TEACHER_MYSQL_PASSWORD", "MYSQL_PASSWORD"),
        database=_mysql_env("DEER_FLOW_TEACHER_MYSQL_DB", "MYSQL_DATABASE", "MYSQL_DB"),
        charset="utf8mb4",
        autocommit=True,
        cursorclass=DictCursor,
    )


_DIFFICULTY_ALIASES = {
    "简单": "简单",
    "中等": "中等",
    "困难": "困难",
    "easy": "简单",
    "medium": "中等",
    "hard": "困难",
    "easy_level": "简单",
    "medium_level": "中等",
    "hard_level": "困难",
}

_SUBJECT_ALIASES = {
    "数学": "数学",
    "math": "数学",
    "mathematics": "数学",
    "语文": "语文",
    "chinese": "语文",
    "english": "英语",
    "英语": "英语",
    "physics": "物理",
    "物理": "物理",
    "chemistry": "化学",
    "化学": "化学",
    "biology": "生物",
    "生物": "生物",
}

_PROBLEM_TYPE_ALIASES = {
    "单选": "单选",
    "single_choice": "单选",
    "single choice": "单选",
    "单选题": "单选",
    "多选": "多选",
    "multiple_choice": "多选",
    "multiple choice": "多选",
    "多选题": "多选",
    "填空": "填空",
    "blank": "填空",
    "fill_in_blank": "填空",
    "fill in the blank": "填空",
    "填空题": "填空",
    "大题": "大题",
    "subjective": "大题",
    "essay": "大题",
    "解答题": "大题",
    "主观题": "大题",
}


def _normalize_string(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def normalize_difficulty(value: Any) -> str | None:
    text = _normalize_string(value)
    if text is None:
        return None
    return _DIFFICULTY_ALIASES.get(text.casefold(), _DIFFICULTY_ALIASES.get(text, text if text in {"简单", "中等", "困难"} else None))


def normalize_subject(value: Any) -> str | None:
    text = _normalize_string(value)
    if text is None:
        return None
    return _SUBJECT_ALIASES.get(text.casefold(), _SUBJECT_ALIASES.get(text, text if text in set(_SUBJECT_ALIASES.values()) else None))


def normalize_problem_type(value: Any) -> str | None:
    text = _normalize_string(value)
    if text is None:
        return None
    return _PROBLEM_TYPE_ALIASES.get(text.casefold(), _PROBLEM_TYPE_ALIASES.get(text, text if text in set(_PROBLEM_TYPE_ALIASES.values()) else None))


def _normalize_list(values: Sequence[Any] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _json_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _normalize_list(value)
    text = str(value).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _normalize_list(text.replace("，", ",").split(","))
    return _normalize_list(data if isinstance(data, list) else [])


def _content_hash(question: str) -> str:
    normalized = "".join(ch for ch in question if not ch.isspace() and ch not in "，。,.？！?!：:；;、")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _student_weakness_terms(student_id: str | None) -> set[str]:
    if not student_id:
        return set()
    markdown = read_student_profile_summary(student_id)
    if not markdown:
        return set()
    profile = parse_student_profile_markdown(student_id, markdown)
    terms: set[str] = set()
    for key in ("weak_knowledge", "weak_ability"):
        for item in profile.get(key) or []:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
            else:
                name = str(item).strip()
            if name:
                terms.add(name)
    return terms


def _chinese_chars(text: str) -> set[str]:
    return {char for char in text if "一" <= char <= "鿿"}


def _similarity_score(source: str, target: str) -> tuple[int, int, int]:
    source_chars = _chinese_chars(source)
    target_chars = _chinese_chars(target)
    overlap = len(source_chars & target_chars)
    contains = int(target in source or source in target)
    return contains, overlap, -len(target)


def _match_known_knowledge_detail(text: str) -> tuple[str | None, str | None]:
    best_match: tuple[tuple[int, int, int], str, str] | None = None
    for knowledge_type, details in KNOWLEDGE_DETAIL_MAP.items():
        for detail in details:
            score = _similarity_score(text, detail)
            if score[0] == 0 and score[1] == 0:
                continue
            if best_match is None or score > best_match[0]:
                best_match = (score, knowledge_type, detail)
    if best_match is None:
        return None, None
    return best_match[1], best_match[2]


def normalize_knowledge_pair(value_type: Any, value_detail: Any, knowledges: Sequence[Any] | None = None) -> tuple[str | None, str | None]:
    candidates = []
    normalized_knowledges = _normalize_list(knowledges)
    for value in (value_detail, value_type, *normalized_knowledges):
        text = _normalize_string(value)
        if text and any("\u4e00" <= ch <= "\u9fff" for ch in text):
            candidates.append(text)
    best_type: str | None = None
    best_detail: str | None = None
    best_score: tuple[int, int, int] | None = None
    for candidate in candidates:
        knowledge_type, detail = _match_known_knowledge_detail(candidate)
        if not knowledge_type or not detail:
            if candidate in MATH_KNOWLEDGE_TYPES and best_type is None:
                best_type = candidate
            continue
        score = _similarity_score(candidate, detail)
        if best_score is None or score > best_score:
            best_score = score
            best_type = knowledge_type
            best_detail = detail
    if best_type and best_detail:
        return best_type, best_detail
    if best_type:
        fallback_detail = KNOWLEDGE_DETAIL_MAP.get(best_type, ())
        return best_type, fallback_detail[0] if fallback_detail else None
    return None, None


def normalize_knowledge_tags(
    value_type: Any,
    value_detail: Any,
    knowledges: Sequence[Any] | None = None,
) -> tuple[str | None, str | None, list[str]]:
    knowledge_type, knowledge_detail = normalize_knowledge_pair(value_type, value_detail, knowledges)
    normalized_details: list[str] = []
    seen: set[str] = set()
    if knowledge_detail:
        normalized_details.append(knowledge_detail)
        seen.add(knowledge_detail.casefold())
    for item in _normalize_list(knowledges):
        item_type, item_detail = normalize_knowledge_pair(None, item, None)
        if knowledge_type is None and item_type is not None:
            knowledge_type = item_type
        if item_detail is None:
            continue
        key = item_detail.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized_details.append(item_detail)
    if knowledge_type is None and normalized_details:
        knowledge_type, _ = normalize_knowledge_pair(None, normalized_details[0], None)
    if knowledge_detail is None and normalized_details:
        knowledge_detail = normalized_details[0]
    if knowledge_type and not normalized_details:
        fallback_detail = KNOWLEDGE_DETAIL_MAP.get(knowledge_type, ())
        if fallback_detail:
            knowledge_detail = fallback_detail[0]
            normalized_details = [knowledge_detail]
    return knowledge_type, knowledge_detail, normalized_details


def normalize_knowledge_type(value: Any, knowledges: Sequence[Any] | None = None) -> str | None:
    knowledge_type, _knowledge_detail = normalize_knowledge_pair(value, None, knowledges)
    return knowledge_type


def normalize_knowledge_detail(value: Any, knowledges: Sequence[Any] | None = None) -> str | None:
    _knowledge_type, knowledge_detail, normalized_details = normalize_knowledge_tags(None, value, knowledges)
    if knowledge_detail:
        return knowledge_detail
    normalized = _normalize_list(normalized_details)
    return ", ".join(normalized) if normalized else None


def _ensure_mysql_schema(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_MYSQL_TABLE} (
                qid BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                sid VARCHAR(128) NULL,
                content TEXT NOT NULL,
                type VARCHAR(64) NULL,
                date DATETIME(6) NOT NULL,
                subject VARCHAR(64) NULL,
                grade VARCHAR(64) NULL,
                stage VARCHAR(32) NULL,
                knowledgeType TEXT NULL,
                difficulty VARCHAR(32) NULL,
                knowledgeDetail TEXT NULL,
                abilityTags TEXT NULL,
                methodTags TEXT NULL,
                errorTags TEXT NULL,
                qualityScore DOUBLE NULL,
                usageCount INT NOT NULL DEFAULT 1,
                successRate DOUBLE NULL,
                source VARCHAR(128) NULL,
                contentHash CHAR(64) NULL,
                status VARCHAR(32) NULL,
                PRIMARY KEY (qid),
                KEY idx_sid_date (sid, date),
                KEY idx_subject (subject)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        cursor.execute(f"SHOW COLUMNS FROM {_MYSQL_TABLE}")
        columns = {row["Field"] for row in cursor.fetchall()}
        column_defs = {
            "grade": "VARCHAR(64) NULL",
            "stage": "VARCHAR(32) NULL",
            "difficulty": "VARCHAR(32) NULL",
            "knowledgeDetail": "TEXT NULL",
            "abilityTags": "TEXT NULL",
            "methodTags": "TEXT NULL",
            "errorTags": "TEXT NULL",
            "qualityScore": "DOUBLE NULL",
            "usageCount": "INT NOT NULL DEFAULT 1",
            "successRate": "DOUBLE NULL",
            "source": "VARCHAR(128) NULL",
            "contentHash": "CHAR(64) NULL",
            "status": "VARCHAR(32) NULL",
        }
        for column, definition in column_defs.items():
            if column not in columns:
                cursor.execute(f"ALTER TABLE {_MYSQL_TABLE} ADD COLUMN {column} {definition}")


def _mongo_collection():
    from pymongo import MongoClient

    client = MongoClient(_mysql_env("DEER_FLOW_TEACHER_MONGO_URI", "MONGODB_URI", "MONGO_URI"), tz_aware=True)
    return client, client[_mysql_env("DEER_FLOW_TEACHER_MONGO_DB", "MONGODB_DATABASE", "MONGO_DB")][_MONGO_COLLECTION]


def _ensure_mongo_indexes(collection) -> None:
    collection.create_index("qid", unique=True, sparse=True)


def _fallback_problem_id() -> int:
    return int(time.time() * 1000)


def _is_math_subject(subject: str | None) -> bool:
    normalized = normalize_subject(subject)
    if normalized is None:
        return True
    return normalized == "数学"


_DIFFICULTY_ORDER = {"简单": 1, "中等": 2, "困难": 3}


def _overlap_count(source: Sequence[str], target: Sequence[str]) -> int:
    return len(set(source) & set(target))


def _difficulty_fit(difficulty: str | None, target_difficulty: str | None) -> int:
    if not difficulty or not target_difficulty:
        return 0
    delta = abs(_DIFFICULTY_ORDER.get(difficulty, 0) - _DIFFICULTY_ORDER.get(target_difficulty, 0))
    if delta == 0:
        return 1
    if delta == 1:
        return 0
    return -1


def _score_candidate(
    *,
    content: str,
    question: str,
    grade: str | None,
    target_grade: str | None,
    stage: str | None,
    target_stage: str | None,
    difficulty: str | None,
    target_difficulty: str | None,
    knowledge_type: str | None,
    target_knowledge_type: str | None,
    knowledge_detail: str | None,
    target_knowledge_detail: str | None,
    ability_tags: Sequence[str],
    target_ability_tags: Sequence[str],
    method_tags: Sequence[str],
    target_method_tags: Sequence[str],
    error_tags: Sequence[str],
    target_error_tags: Sequence[str],
    quality_score: float | None,
    status: str | None,
    student_weakness_terms: set[str],
    created_at: Any,
) -> tuple[int, float]:
    if content.strip() == question.strip():
        return -100, 0.0
    score = 0
    if knowledge_detail and target_knowledge_detail and knowledge_detail == target_knowledge_detail:
        score += 12
    elif knowledge_detail and target_knowledge_detail and any(item in knowledge_detail for item in target_knowledge_detail.split(", ")):
        score += 8
    if knowledge_type and target_knowledge_type and knowledge_type == target_knowledge_type:
        score += 6
    score += 5 * _overlap_count(error_tags, target_error_tags)
    score += 4 * _overlap_count(method_tags, target_method_tags)
    score += 4 * _overlap_count(ability_tags, target_ability_tags)
    score += 3 * _difficulty_fit(difficulty, target_difficulty)
    if grade and target_grade and grade == target_grade:
        score += 2
    elif stage and target_stage and stage == target_stage:
        score += 2
    if quality_score is not None:
        score += min(2, max(0, int(quality_score)))
    if status == "reviewed":
        score += 2
    candidate_terms = {item for item in [knowledge_detail, knowledge_type, *ability_tags, *method_tags, *error_tags] if item}
    if candidate_terms & student_weakness_terms:
        score += 2
    timestamp = 0.0
    if isinstance(created_at, datetime):
        timestamp = created_at.timestamp()
    return score, timestamp


def _recommend_reason(
    *,
    knowledge_detail: str | None,
    target_knowledge_detail: str | None,
    difficulty: str | None,
    target_difficulty: str | None,
    ability_tags: Sequence[str],
    target_ability_tags: Sequence[str],
    method_tags: Sequence[str],
    target_method_tags: Sequence[str],
    error_tags: Sequence[str],
    target_error_tags: Sequence[str],
) -> str:
    reasons: list[str] = []
    if knowledge_detail and target_knowledge_detail and knowledge_detail == target_knowledge_detail:
        reasons.append(f"同考{knowledge_detail}")
    ability_overlap = set(ability_tags) & set(target_ability_tags)
    method_overlap = set(method_tags) & set(target_method_tags)
    error_overlap = set(error_tags) & set(target_error_tags)
    if method_overlap:
        reasons.append(f"方法相近：{'、'.join(sorted(method_overlap))}")
    if ability_overlap:
        reasons.append(f"能力训练相近：{'、'.join(sorted(ability_overlap))}")
    if error_overlap:
        reasons.append(f"适合巩固错因：{'、'.join(sorted(error_overlap))}")
    if difficulty and target_difficulty:
        if difficulty == target_difficulty:
            reasons.append("难度相同，适合同类巩固")
        else:
            reasons.append(f"难度从{target_difficulty}调整到{difficulty}，适合分层练习")
    return "；".join(reasons) or "基于题库元信息匹配的同类练习。"


def retrieve_similar_problems(
    *,
    question: str,
    student_id: str | None,
    subject: str | None,
    knowledges: Sequence[Any] | None,
    difficulty: str | None,
    knowledge_type: str | None = None,
    knowledge_detail: str | None = None,
    grade: str | None = None,
    stage: str | None = None,
    ability_tags: Sequence[Any] | None = None,
    method_tags: Sequence[Any] | None = None,
    error_tags: Sequence[Any] | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    subject = normalize_subject(subject)
    target_knowledge_type = normalize_knowledge_type(knowledge_type, knowledges)
    target_knowledge_detail = normalize_knowledge_detail(knowledge_detail, knowledges)
    difficulty = normalize_difficulty(difficulty)
    stage = normalize_stage(stage, grade)
    target_ability_tags = normalize_ability_tags(ability_tags)
    target_method_tags = normalize_method_tags(method_tags)
    target_error_tags = normalize_error_tags(error_tags)
    student_weakness_terms = _student_weakness_terms(student_id)
    connection = _mysql_connection()
    try:
        _ensure_mysql_schema(connection)
        with connection.cursor() as cursor:
            conditions = ["subject = %s"] if subject else []
            params: list[Any] = [subject] if subject else []
            if target_knowledge_type:
                conditions.append("knowledgeType = %s")
                params.append(target_knowledge_type)
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            cursor.execute(
                f"""
                SELECT qid, content, type, date, subject, grade, stage, knowledgeType, difficulty, knowledgeDetail, abilityTags, methodTags, errorTags, qualityScore, status
                FROM {_MYSQL_TABLE}
                {where_clause}
                ORDER BY date DESC
                LIMIT 100
                """,
                tuple(params),
            )
            basic_rows = cursor.fetchall()
    finally:
        connection.close()

    if not basic_rows:
        return []

    detail_map: dict[int, dict[str, Any]] = {}
    client = None
    try:
        client, collection = _mongo_collection()
        qids = [row["qid"] for row in basic_rows]
        for document in collection.find({"qid": {"$in": qids}}):
            detail_map[document["qid"]] = document
    finally:
        if client is not None:
            client.close()

    scored: list[tuple[tuple[int, float], dict[str, Any]]] = []
    for row in basic_rows:
        row_ability_tags = _parse_json_list(row.get("abilityTags"))
        row_method_tags = _parse_json_list(row.get("methodTags"))
        row_error_tags = _parse_json_list(row.get("errorTags"))
        score = _score_candidate(
            content=row.get("content") or "",
            question=question,
            grade=row.get("grade"),
            target_grade=grade,
            stage=row.get("stage"),
            target_stage=stage,
            difficulty=row.get("difficulty"),
            target_difficulty=difficulty,
            knowledge_type=row.get("knowledgeType"),
            target_knowledge_type=target_knowledge_type,
            knowledge_detail=row.get("knowledgeDetail"),
            target_knowledge_detail=target_knowledge_detail,
            ability_tags=row_ability_tags,
            target_ability_tags=target_ability_tags,
            method_tags=row_method_tags,
            target_method_tags=target_method_tags,
            error_tags=row_error_tags,
            target_error_tags=target_error_tags,
            quality_score=row.get("qualityScore"),
            status=row.get("status"),
            student_weakness_terms=student_weakness_terms,
            created_at=row.get("date"),
        )
        if score[0] < 0:
            continue
        detail = detail_map.get(row["qid"], {})
        scored.append(
            (
                score,
                {
                    "qid": row["qid"],
                    "question": detail.get("question") or row.get("content") or "",
                    "answer": detail.get("answer") or "",
                    "explanation": detail.get("explanation") or "",
                    "steps": detail.get("steps") or [],
                    "knowledges": detail.get("knowledges") or [],
                    "problem_type": detail.get("problem_type") or row.get("type"),
                    "difficulty": detail.get("difficulty") or row.get("difficulty"),
                    "knowledge_type": row.get("knowledgeType"),
                    "knowledge_detail": row.get("knowledgeDetail"),
                    "subject": row.get("subject"),
                    "grade": detail.get("grade") or row.get("grade"),
                    "stage": row.get("stage"),
                    "ability_tags": detail.get("ability_tags") or row_ability_tags,
                    "method_tags": detail.get("method_tags") or row_method_tags,
                    "error_tags": detail.get("error_tags") or row_error_tags,
                    "recommend_reason": _recommend_reason(
                        knowledge_detail=row.get("knowledgeDetail"),
                        target_knowledge_detail=target_knowledge_detail,
                        difficulty=row.get("difficulty"),
                        target_difficulty=difficulty,
                        ability_tags=row_ability_tags,
                        target_ability_tags=target_ability_tags,
                        method_tags=row_method_tags,
                        target_method_tags=target_method_tags,
                        error_tags=row_error_tags,
                        target_error_tags=target_error_tags,
                    ),
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def persist_generated_problem_result(
    *,
    question: str,
    student_id: str | None,
    image_url: str | None,
    subject: str | None,
    grade: str | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    persisted_at = _utc_now()
    student_id = student_id or _DEFAULT_STUDENT_ID
    problem_id: int | None = None
    detail_id: str | None = None
    markdown_path: str | None = None
    subject = normalize_subject(subject)
    problem_type = normalize_problem_type(result.get("problem_type"))
    raw_knowledges = _normalize_list(result.get("knowledges") or [])
    knowledge_type, knowledge_detail, knowledges = normalize_knowledge_tags(result.get("knowledge_type"), result.get("knowledge_detail"), raw_knowledges)
    difficulty = normalize_difficulty(result.get("difficulty"))
    stage = normalize_stage(result.get("stage"), grade)
    ability_tags = normalize_ability_tags(result.get("ability_tags"))
    method_tags = normalize_method_tags(result.get("method_tags"))
    error_tags = normalize_error_tags(result.get("error_tags"))
    quality_score = result.get("quality_score")
    success_rate = result.get("success_rate")
    source = _normalize_string(result.get("source")) or "digital-teacher"
    status = _normalize_string(result.get("status")) or "auto_labeled"
    problem_id = _fallback_problem_id()
    content_hash = _content_hash(question)

    if _mysql_enabled():
        connection = _mysql_connection()
        try:
            _ensure_mysql_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {_MYSQL_TABLE} (
                        qid,
                        sid,
                        content,
                        type,
                        date,
                        subject,
                        grade,
                        stage,
                        knowledgeType,
                        difficulty,
                        knowledgeDetail,
                        abilityTags,
                        methodTags,
                        errorTags,
                        qualityScore,
                        usageCount,
                        successRate,
                        source,
                        contentHash,
                        status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        problem_id,
                        student_id,
                        question,
                        problem_type,
                        persisted_at,
                        subject,
                        grade,
                        stage,
                        knowledge_type,
                        difficulty,
                        knowledge_detail,
                        _json_list(ability_tags),
                        _json_list(method_tags),
                        _json_list(error_tags),
                        quality_score,
                        int(result.get("usage_count") or 1),
                        success_rate,
                        source,
                        content_hash,
                        status,
                    ),
                )
        finally:
            connection.close()

    if _mongo_enabled():
        client, collection = _mongo_collection()
        try:
            _ensure_mongo_indexes(collection)
            document = {
                "qid": problem_id,
                "student_id": student_id,
                "subject": subject,
                "grade": grade,
                "created_at": persisted_at,
                "question": question,
                "answer": result.get("answer"),
                "explanation": result.get("explanation"),
                "steps": result.get("steps") or [],
                "knowledges": knowledges,
                "problem_type": problem_type,
                "difficulty": difficulty,
                "stage": stage,
                "knowledge_type": knowledge_type,
                "knowledge_detail": knowledge_detail,
                "ability_tags": ability_tags,
                "method_tags": method_tags,
                "error_tags": error_tags,
                "quality_score": quality_score,
                "success_rate": success_rate,
                "source": source,
                "content_hash": content_hash,
                "status": status,
                "error_analysis": result.get("error_analysis"),
                "common_mistakes": result.get("common_mistakes") or [],
                "solution_methods": result.get("solution_methods") or [],
                "raw_tags": result.get("raw_tags") or [],
                "weak_knowledge_candidates": result.get("weak_knowledge_candidates") or [],
                "weak_ability_candidates": result.get("weak_ability_candidates") or [],
                "original_artifact": {
                    "image_url": image_url,
                    "oss_uri": None,
                },
            }
            inserted = collection.insert_one(document)
            detail_id = str(inserted.inserted_id)
            logger.info("teacher Mongo persistence succeeded: qid=%s detail_id=%s collection=%s", problem_id, detail_id, _MONGO_COLLECTION)
        finally:
            client.close()

    if student_id and _is_math_subject(subject):
        markdown_path = str(
            update_student_profile_from_observation(
                student_id,
                observed_at=persisted_at,
                problem_id=problem_id,
                subject=subject,
                grade=grade,
                knowledges=knowledges or None,
                weak_knowledge=result.get("weak_knowledge_candidates") or None,
                weak_ability=result.get("weak_ability_candidates") or None,
                summary=result.get("explanation") or None,
                problem_type=problem_type,
                difficulty=difficulty,
                error_analysis=result.get("error_analysis"),
            )
        )

    return {
        "problem_id": problem_id,
        "problem_detail_id": detail_id,
        "student_profile_path": markdown_path,
        "created_at": persisted_at.isoformat(),
    }


def persist_safely(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return persist_generated_problem_result(**kwargs), None
    except (OSError, ValueError, ImportError, ModuleNotFoundError) as exc:
        return None, str(exc)
    except Exception as exc:
        name = exc.__class__.__module__
        if name.startswith("pymysql") or name.startswith("pymongo"):
            return None, f"{exc.__class__.__name__}: {exc}"
        raise


async def persist_safely_async(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    return await asyncio.to_thread(persist_safely, **kwargs)
