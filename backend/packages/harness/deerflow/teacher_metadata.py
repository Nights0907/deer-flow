from __future__ import annotations

ABILITY_TAGS = (
    "计算能力",
    "逻辑推理",
    "抽象概括",
    "数形结合",
    "建模能力",
    "分类讨论",
    "转化化归",
    "空间想象",
    "数据分析",
    "阅读理解",
)

METHOD_TAGS = (
    "待定系数法",
    "换元法",
    "配方法",
    "因式分解",
    "构造函数",
    "分类讨论法",
    "数形结合法",
    "参数法",
    "反证法",
    "数学归纳法",
    "导数法",
    "坐标法",
    "辅助线法",
)

ERROR_TAGS = (
    "概念混淆",
    "公式误用",
    "漏条件",
    "计算错误",
    "审题错误",
    "分类不全",
    "符号错误",
    "单位错误",
    "图像理解错误",
    "步骤跳跃",
    "结论未验证",
)

_STAGE_ALIASES = {
    "小学": "小学",
    "primary": "小学",
    "初中": "初中",
    "middle": "初中",
    "junior": "初中",
    "高中": "高中",
    "high": "高中",
    "senior": "高中",
}

_GRADE_STAGE_MAP = {
    "一年级": "小学",
    "二年级": "小学",
    "三年级": "小学",
    "四年级": "小学",
    "五年级": "小学",
    "六年级": "小学",
    "七年级": "初中",
    "八年级": "初中",
    "九年级": "初中",
    "初一": "初中",
    "初二": "初中",
    "初三": "初中",
    "高一": "高中",
    "高二": "高中",
    "高三": "高中",
}


def normalize_stage(value: object, grade: object | None = None) -> str | None:
    text = str(value).strip() if value is not None else ""
    if text:
        normalized = _STAGE_ALIASES.get(text.casefold(), _STAGE_ALIASES.get(text))
        if normalized:
            return normalized
    grade_text = str(grade).strip() if grade is not None else ""
    for key, stage in _GRADE_STAGE_MAP.items():
        if key in grade_text:
            return stage
    return None


def _normalize_tag_list(values: object, allowed_tags: tuple[str, ...]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        candidates = [item.strip() for item in values.replace("，", ",").split(",")]
    elif isinstance(values, (list, tuple, set)):
        candidates = [str(item).strip() for item in values]
    else:
        candidates = [str(values).strip()]
    allowed = set(allowed_tags)
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        matched = candidate if candidate in allowed else next((tag for tag in allowed_tags if tag in candidate or candidate in tag), None)
        if matched is None or matched in seen:
            continue
        seen.add(matched)
        normalized.append(matched)
    return normalized


def normalize_ability_tags(values: object) -> list[str]:
    return _normalize_tag_list(values, ABILITY_TAGS)


def normalize_method_tags(values: object) -> list[str]:
    return _normalize_tag_list(values, METHOD_TAGS)


def normalize_error_tags(values: object) -> list[str]:
    return _normalize_tag_list(values, ERROR_TAGS)
