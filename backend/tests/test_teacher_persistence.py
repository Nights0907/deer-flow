import json
from pathlib import Path

from deerflow import teacher_persistence


class FakeMysqlCursor:
    def __init__(self, connection):
        self.connection = connection
        self.lastrowid = 321
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.connection.executed.append((sql, params))
        if "SHOW COLUMNS" in sql:
            self._result = [{"Field": field} for field in self.connection.columns]
        elif "SELECT qid, content" in sql:
            self._result = list(self.connection.select_rows)
        else:
            self._result = []

    def fetchall(self):
        return list(self._result)


class FakeMysqlConnection:
    def __init__(self, columns=None, select_rows=None):
        self.executed = []
        self.closed = False
        self.columns = columns or ["qid", "sid", "content", "type", "date", "subject", "knowledgeType"]
        self.select_rows = select_rows or []

    def cursor(self):
        return FakeMysqlCursor(self)

    def close(self):
        self.closed = True


class FakeInsertResult:
    inserted_id = "mongo-1"


class FakeMongoCollection:
    def __init__(self):
        self.indexes = []
        self.documents = []
        self.find_result = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def insert_one(self, document):
        self.documents.append(document)
        return FakeInsertResult()

    def find(self, _query):
        return list(self.find_result)


class FakeMongoClient:
    def __init__(self, collection):
        self.collection = collection
        self.closed = False

    def __getitem__(self, _db_name):
        collection = self.collection

        class FakeMongoDatabase:
            def __getitem__(self, _collection_name):
                return collection

        return FakeMongoDatabase()

    def close(self):
        self.closed = True


def test_persist_generated_problem_result_writes_markdown_only_when_databases_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(teacher_persistence, "_mysql_enabled", lambda: False)
    monkeypatch.setattr(teacher_persistence, "_mongo_enabled", lambda: False)
    monkeypatch.setattr(teacher_persistence, "_fallback_problem_id", lambda: 555)
    monkeypatch.setattr(teacher_persistence, "update_student_profile_from_observation", lambda student_id, **kwargs: tmp_path / student_id / "PROFILE.md")

    result = teacher_persistence.persist_generated_problem_result(
        question="1+1?",
        student_id="stu-1",
        image_url=None,
        subject="数学",
        grade="grade-1",
        result={
            "answer": "2",
            "steps": ["count"],
            "explanation": "add one and one",
            "knowledges": ["addition"],
            "error_analysis": "counting mistake",
            "weak_knowledge_candidates": ["addition within 10"],
            "weak_ability_candidates": ["careful checking"],
            "raw": {"core": {"answer": "2"}},
        },
    )

    assert result["problem_id"] == 555
    assert result["problem_detail_id"] is None
    assert result["student_profile_path"] == str(tmp_path / "stu-1" / "PROFILE.md")
    assert result["created_at"]


def test_persist_safely_returns_error_string(monkeypatch):
    monkeypatch.setattr(teacher_persistence, "persist_generated_problem_result", lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    persisted, error = teacher_persistence.persist_safely(question="q", student_id=None, image_url=None, subject=None, grade=None, result={})

    assert persisted is None
    assert error == "disk full"


def test_persist_generated_problem_result_defaults_student_id_for_mysql(monkeypatch):
    mysql = FakeMysqlConnection()
    monkeypatch.setattr(teacher_persistence, "_mysql_enabled", lambda: True)
    monkeypatch.setattr(teacher_persistence, "_mongo_enabled", lambda: False)
    monkeypatch.setattr(teacher_persistence, "_mysql_connection", lambda: mysql)
    monkeypatch.setattr(teacher_persistence, "_fallback_problem_id", lambda: 999)
    monkeypatch.setattr(teacher_persistence, "update_student_profile_from_observation", lambda student_id, **kwargs: Path("/tmp/default/PROFILE.md"))

    teacher_persistence.persist_generated_problem_result(
        question="1+1?",
        student_id=None,
        image_url=None,
        subject="数学",
        grade="grade-1",
        result={"answer": "2", "steps": [], "explanation": "ok", "knowledges": []},
    )

    insert_sql, insert_params = next((sql, params) for sql, params in mysql.executed if sql.strip().startswith("INSERT INTO"))
    assert "INSERT INTO question_basic_info" in insert_sql
    assert insert_params[0] == 999
    assert insert_params[1] == "522025320226"


def test_mysql_enabled_defaults_to_true_without_env(monkeypatch):
    monkeypatch.delenv("DEER_FLOW_TEACHER_MYSQL_HOST", raising=False)
    monkeypatch.delenv("DEER_FLOW_TEACHER_MYSQL_DB", raising=False)
    monkeypatch.delenv("MYSQL_HOST", raising=False)
    monkeypatch.delenv("MYSQL_DATABASE", raising=False)

    assert teacher_persistence._mysql_enabled() is True


def test_mongo_enabled_defaults_to_true_without_env(monkeypatch):
    monkeypatch.delenv("DEER_FLOW_TEACHER_MONGO_URI", raising=False)
    monkeypatch.delenv("DEER_FLOW_TEACHER_MONGO_DB", raising=False)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DATABASE", raising=False)

    assert teacher_persistence._mongo_enabled() is True


def test_mysql_and_mongo_defaults_are_available_without_env(monkeypatch):
    monkeypatch.delenv("MYSQL_HOST", raising=False)
    monkeypatch.delenv("MYSQL_DATABASE", raising=False)
    monkeypatch.delenv("MYSQL_USER", raising=False)
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DATABASE", raising=False)

    assert teacher_persistence._mysql_env("MYSQL_HOST") == "127.0.0.1"
    assert teacher_persistence._mysql_env("MYSQL_DATABASE") == "education"
    assert teacher_persistence._mysql_env("MYSQL_USER") == "root"
    assert teacher_persistence._mysql_env("MYSQL_PASSWORD") == "123456"
    assert teacher_persistence._mysql_env("MONGODB_URI") == "mongodb://127.0.0.1:27017"
    assert teacher_persistence._mysql_env("MONGODB_DATABASE") == "education"


def test_persist_generated_problem_result_ensures_mysql_schema_and_mongo_indexes(monkeypatch, tmp_path):
    mysql = FakeMysqlConnection()
    mongo_collection = FakeMongoCollection()
    mongo_client = FakeMongoClient(mongo_collection)
    captured = {}

    def fake_update_profile(student_id, **kwargs):
        captured["student_id"] = student_id
        captured.update(kwargs)
        return tmp_path / student_id / "PROFILE.md"

    monkeypatch.setattr(teacher_persistence, "_mysql_enabled", lambda: True)
    monkeypatch.setattr(teacher_persistence, "_mongo_enabled", lambda: True)
    monkeypatch.setattr(teacher_persistence, "_mysql_connection", lambda: mysql)
    monkeypatch.setattr(teacher_persistence, "_mongo_collection", lambda: (mongo_client, mongo_collection))
    monkeypatch.setattr(teacher_persistence, "_fallback_problem_id", lambda: 321)
    monkeypatch.setattr(teacher_persistence, "update_student_profile_from_observation", fake_update_profile)

    result = teacher_persistence.persist_generated_problem_result(
        question="1+1?",
        student_id="stu-1",
        image_url=None,
        subject="数学",
        grade="grade-1",
        result={
            "answer": "2",
            "steps": ["count"],
            "explanation": "add one and one",
            "knowledges": ["monotonicity", "函数单调综合"],
            "knowledge_type": "function",
            "knowledge_detail": "单调",
            "ability_tags": ["数形结合"],
            "method_tags": ["构造函数"],
            "error_tags": ["公式误用"],
            "weak_knowledge_candidates": ["addition within 10"],
            "weak_ability_candidates": ["careful checking"],
        },
    )

    assert result["problem_id"] == 321
    assert result["problem_detail_id"] == "mongo-1"
    create_sql, _ = mysql.executed[0]
    insert_sql, insert_params = next((sql, params) for sql, params in mysql.executed if sql.strip().startswith("INSERT INTO"))
    assert "CREATE TABLE IF NOT EXISTS question_basic_info" in create_sql
    assert "qid BIGINT UNSIGNED NOT NULL AUTO_INCREMENT" in create_sql
    assert "sid VARCHAR(128) NULL" in create_sql
    assert "knowledgeType TEXT NULL" in create_sql
    assert "difficulty VARCHAR(32) NULL" in create_sql
    assert "knowledgeDetail TEXT NULL" in create_sql
    assert "INSERT INTO question_basic_info" in insert_sql
    assert "qid" in insert_sql and "sid" in insert_sql and "content" in insert_sql and "knowledgeType" in insert_sql
    assert insert_params[0] == 321
    assert insert_params[1] == "stu-1"
    assert insert_params[2] == "1+1?"
    assert insert_params[3] is None
    assert insert_params[5] == "数学"
    assert insert_params[6] == "grade-1"
    assert insert_params[7] is None
    assert insert_params[8] == "函数"
    assert insert_params[9] is None
    assert insert_params[10] == "函数的单调性"
    assert insert_params[11] == '["数形结合"]'
    assert insert_params[12] == '["构造函数"]'
    assert insert_params[13] == '["公式误用"]'
    assert mongo_collection.indexes == [
        ("qid", {"unique": True, "sparse": True}),
    ]
    assert mongo_collection.documents[0] == {
        "qid": 321,
        "student_id": "stu-1",
        "subject": "数学",
        "grade": "grade-1",
        "created_at": mongo_collection.documents[0]["created_at"],
        "question": "1+1?",
        "answer": "2",
        "explanation": "add one and one",
        "steps": ["count"],
        "knowledges": ["函数的单调性"],
        "problem_type": None,
        "difficulty": None,
        "stage": None,
        "knowledge_type": "函数",
        "knowledge_detail": "函数的单调性",
        "ability_tags": ["数形结合"],
        "method_tags": ["构造函数"],
        "error_tags": ["公式误用"],
        "quality_score": None,
        "success_rate": None,
        "source": "digital-teacher",
        "content_hash": mongo_collection.documents[0]["content_hash"],
        "status": "auto_labeled",
        "error_analysis": None,
        "common_mistakes": [],
        "solution_methods": [],
        "raw_tags": [],
        "weak_knowledge_candidates": ["addition within 10"],
        "weak_ability_candidates": ["careful checking"],
        "original_artifact": {"image_url": None, "oss_uri": None},
    }
    assert mysql.closed is True
    assert mongo_client.closed is True
    assert captured["student_id"] == "stu-1"
    assert captured["problem_id"] == 321
    assert captured["subject"] == "数学"
    assert captured["grade"] == "grade-1"
    assert captured["knowledges"] == ["函数的单调性"]
    assert captured["weak_knowledge"] == ["addition within 10"]
    assert captured["weak_ability"] == ["careful checking"]
    assert captured["summary"] == "add one and one"
    assert captured["problem_type"] is None
    assert captured["difficulty"] is None
    assert captured["error_analysis"] is None
    assert captured["observed_at"]


def test_persist_generated_problem_result_skips_profile_update_for_non_math_subject(monkeypatch):
    monkeypatch.setattr(teacher_persistence, "_mysql_enabled", lambda: False)
    monkeypatch.setattr(teacher_persistence, "_mongo_enabled", lambda: False)
    monkeypatch.setattr(teacher_persistence, "_fallback_problem_id", lambda: 777)
    called = {"value": False}

    def fake_update_profile(student_id, **kwargs):
        called["value"] = True
        return Path("/tmp/PROFILE.md")

    monkeypatch.setattr(teacher_persistence, "update_student_profile_from_observation", fake_update_profile)

    result = teacher_persistence.persist_generated_problem_result(
        question="translate text",
        student_id="stu-1",
        image_url=None,
        subject="english",
        grade="grade-1",
        result={"answer": "ok", "steps": [], "explanation": "ok", "knowledges": []},
    )

    assert called["value"] is False
    assert result["student_profile_path"] is None


def test_ensure_mysql_schema_repairs_missing_columns():
    mysql = FakeMysqlConnection(columns=["qid", "sid", "content", "type", "date", "subject", "knowledgeType"])

    teacher_persistence._ensure_mysql_schema(mysql)

    sqls = [sql for sql, _ in mysql.executed]
    assert any("ADD COLUMN difficulty" in sql for sql in sqls)
    assert any("ADD COLUMN knowledgeDetail" in sql for sql in sqls)
    assert any("ADD COLUMN abilityTags" in sql for sql in sqls)
    assert any("ADD COLUMN methodTags" in sql for sql in sqls)
    assert any("ADD COLUMN errorTags" in sql for sql in sqls)


def test_normalize_knowledge_type_prefers_known_detail_mapping():
    assert teacher_persistence.normalize_knowledge_type(None, ["函数的单调性综合题"]) == "函数"


def test_normalize_knowledge_detail_prefers_known_detail_mapping_and_discards_nonstandard_items():
    assert teacher_persistence.normalize_knowledge_detail(None, ["函数的单调性综合题", "custom monotonicity tag"]) == "函数的单调性"


def test_normalize_knowledge_detail_matches_nearest_chinese_term():
    assert teacher_persistence.normalize_knowledge_detail("单调", None) == "函数的单调性"
    assert teacher_persistence.normalize_knowledge_detail("椭圆离心率", None) == "椭圆"
    assert teacher_persistence.normalize_knowledge_detail("等差求和", None) == "等差数列"


def test_normalize_knowledge_tags_discards_english_and_keeps_fixed_chinese_taxonomy():
    knowledge_type, knowledge_detail, knowledges = teacher_persistence.normalize_knowledge_tags(
        "function",
        "单调",
        ["monotonicity", "函数单调综合"],
    )

    assert knowledge_type == "函数"
    assert knowledge_detail == "函数的单调性"
    assert knowledges == ["函数的单调性"]


def test_normalize_difficulty_maps_to_fixed_chinese_values():
    assert teacher_persistence.normalize_difficulty("easy") == "简单"
    assert teacher_persistence.normalize_difficulty("medium") == "中等"
    assert teacher_persistence.normalize_difficulty("hard") == "困难"
    assert teacher_persistence.normalize_difficulty("简单") == "简单"


def test_normalize_subject_and_problem_type_map_to_fixed_chinese_values():
    assert teacher_persistence.normalize_subject("math") == "数学"
    assert teacher_persistence.normalize_subject("english") == "英语"
    assert teacher_persistence.normalize_problem_type("single_choice") == "单选"
    assert teacher_persistence.normalize_problem_type("multiple choice") == "多选"
    assert teacher_persistence.normalize_problem_type("fill_in_blank") == "填空"
    assert teacher_persistence.normalize_problem_type("essay") == "大题"


def test_teacher_metadata_normalizers_map_to_controlled_tags():
    from deerflow.teacher_metadata import normalize_ability_tags, normalize_error_tags, normalize_method_tags, normalize_stage

    assert normalize_stage(None, "高一") == "高中"
    assert normalize_ability_tags(["需要数形结合思想", "unknown"]) == ["数形结合"]
    assert normalize_method_tags(["构造函数求解"]) == ["构造函数"]
    assert normalize_error_tags(["容易公式误用"]) == ["公式误用"]


def test_retrieve_similar_problems_ranks_structured_matches(monkeypatch):
    mysql = FakeMysqlConnection(
        columns=["qid", "sid", "content", "type", "date", "subject", "grade", "stage", "knowledgeType", "difficulty", "knowledgeDetail", "abilityTags", "methodTags", "errorTags", "qualityScore", "status"],
        select_rows=[
            {
                "qid": 1,
                "content": "题目A",
                "type": "equation",
                "date": teacher_persistence._utc_now(),
                "subject": "数学",
                "grade": "高一",
                "stage": "高中",
                "knowledgeType": "函数",
                "difficulty": "简单",
                "knowledgeDetail": "函数单调性",
                "abilityTags": '["数形结合"]',
                "methodTags": '["构造函数"]',
                "errorTags": '["公式误用"]',
                "qualityScore": 2,
                "status": "reviewed",
            },
            {
                "qid": 2,
                "content": "题目B",
                "type": "equation",
                "date": teacher_persistence._utc_now(),
                "subject": "数学",
                "grade": "高一",
                "stage": "高中",
                "knowledgeType": "函数",
                "difficulty": "困难",
                "knowledgeDetail": "函数图像",
                "abilityTags": '["计算能力"]',
                "methodTags": '["配方法"]',
                "errorTags": '["计算错误"]',
                "qualityScore": 0,
                "status": "auto_labeled",
            },
        ],
    )
    mongo_collection = FakeMongoCollection()
    mongo_collection.find_result = [
        {"qid": 1, "question": "题目A", "difficulty": "简单", "knowledges": ["函数单调性"]},
        {"qid": 2, "question": "题目B", "difficulty": "困难", "knowledges": ["函数图像"]},
    ]
    mongo_client = FakeMongoClient(mongo_collection)
    monkeypatch.setattr(teacher_persistence, "_mysql_connection", lambda: mysql)
    monkeypatch.setattr(teacher_persistence, "_mongo_collection", lambda: (mongo_client, mongo_collection))

    results = teacher_persistence.retrieve_similar_problems(
        question="新题",
        student_id="stu-1",
        subject="数学",
        knowledges=["函数单调性"],
        difficulty="简单",
        knowledge_type="函数",
        knowledge_detail="函数单调性",
        grade="高一",
        ability_tags=["数形结合"],
        method_tags=["构造函数"],
        error_tags=["公式误用"],
    )

    assert [item["qid"] for item in results] == [1, 2]
    assert results[0]["ability_tags"] == ["数形结合"]
    assert "构造函数" in results[0]["recommend_reason"]
