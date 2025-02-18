# -*- coding: utf-8 -*-
"""Database helper function for recording"""
import sqlite3


def create_logging_table(db_name: str) -> None:
    """create table in sqlite3"""
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS logging (
        id TEXT PRIMARY KEY,
        user_input TEXT,
        model_response TEXT,
        score FLOAT,
        suggest_answer TEXT
    )
    """,
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS logging_internal (
        request_id TEXT,
        module_name TEXT,
        method_name TEXT,
        step TEXT,
        context TEXT,
        PRIMARY KEY (request_id, module_name, method_name, step),
        FOREIGN KEY (request_id) REFERENCES logging(id)
    )
    """,
    )
    # db_conn.commit()
    db_conn.close()


def record_user_input(db_name: str, rid: str, user_input: str) -> None:
    """record user input"""
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT INTO logging (id, user_input) VALUES (?, ?)",
        (rid, user_input),
    )
    db_conn.commit()
    db_conn.close()


def record_model_response(db_name: str, rid: str, model_response: str) -> None:
    """record user response"""
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    cursor.execute(
        "UPDATE logging SET model_response = ? WHERE id = ?",
        (model_response, rid),
    )
    db_conn.commit()
    db_conn.close()


def record_suggest_answer(db_name: str, rid: str, suggest_answer: str) -> None:
    """record user feedback"""
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    cursor.execute(
        "UPDATE logging SET suggest_answer = ? WHERE id = ?",
        (suggest_answer, rid),
    )
    db_conn.commit()
    db_conn.close()


def record_score(db_name: str, rid: str, score: float) -> None:
    """record score of response if any"""
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    cursor.execute(
        "UPDATE logging SET score = ? WHERE id = ?",
        (score, rid),
    )
    db_conn.commit()
    db_conn.close()


def record_log(
    db_name: str,
    request_id: str,
    location: str,
    context: str,
) -> None:
    """record intermediate step results"""
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()

    module_name, method_name, step = (
        "internal_module",
        "internal_method",
        "internal_step",
    )
    # a typical message is in the format of "module_name.method_name:step"
    if ":" in location:
        parts = location.split(":")
        location, step = parts[0], parts[1]
    if "." in location:
        parts = location.split(".")
        location, method_name = parts[0], parts[1]
    module_name = location

    # check if the (request_id, module_name, method_name) exists in the table
    row_exist = cursor.execute(
        "SELECT 1 FROM logging_internal WHERE request_id = ? "
        "AND module_name = ? AND method_name = ? AND step = ?",
        (request_id, module_name, method_name, step),
    ).fetchone()
    if row_exist:  # if the row exists, append new context to the old
        old_value = cursor.execute(
            "SELECT context FROM logging_internal WHERE request_id = ? "
            "AND module_name = ? AND method_name = ? AND step = ?",
            (request_id, module_name, method_name, step),
        ).fetchone()[0]
        if old_value is not None:
            context = f"{old_value},\n{context}"
        cursor.execute(
            "UPDATE logging_internal SET context = ? WHERE request_id = ? "
            "AND module_name = ? AND method_name = ? AND step = ?",
            (context, request_id, module_name, method_name, step),
        )
    else:  # otherwise, insert a new row into the table
        cursor.execute(
            "INSERT INTO logging_internal (request_id, module_name, "
            "method_name, step, context) VALUES (?, ?, ?, ?, ?)",
            (request_id, module_name, method_name, step, context),
        )

    db_conn.commit()
    db_conn.close()
