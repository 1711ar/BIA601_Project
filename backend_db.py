
import os
import sqlite3
import json
import datetime as dt
import pandas as pd
from typing import Optional, List, Dict, Any

#  المسارات 
DATA_DIR = os.path.join("data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "metadata.db")


#  دوال مساعدة 
def _ensure_dirs() -> None:
    """يتأكد من وجود مجلدات data/ و uploads/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def _get_connection() -> sqlite3.Connection:
    """فتح اتصال آمن بقاعدة البيانات مع التأكد من الإنشاء."""
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    return conn


#  تهيئة القاعدة 
def init_db() -> None:
    """إنشاء الجداول عند أول تشغيل."""
    conn = _get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            target_column TEXT NOT NULL,
            n_features_total INTEGER NOT NULL,
            n_features_selected INTEGER NOT NULL,
            acc_ga REAL,
            acc_pca REAL,
            acc_skb REAL,
            selected_indices TEXT,     -- قائمة JSON بالميزات المختارة من GA
            params_json TEXT,          -- JSON لأي إعدادات إضافية
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()



init_db()


#  رفع الملفات 
def save_uploaded_file(uploaded_file) -> str:
    """
    يحفظ ملف Streamlit UploadedFile على القرص في uploads/ باسم timestamped.
    يرجع المسار الكامل للملف المحفوظ.
    """
    _ensure_dirs()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{ts}__{uploaded_file.name}"
    out_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        if hasattr(uploaded_file, "getbuffer"):
            data = uploaded_file.getbuffer()
            with open(out_path, "wb") as f:
                f.write(data)
        else:
            with open(out_path, "wb") as f:
                f.write(uploaded_file.read())
    except Exception as e:
        raise IOError(f"فشل حفظ الملف: {e}")

    return out_path


# تسجيل التجارب 
def log_run(
    *,
    filename: str,
    stored_path: str,
    target_column: str,
    n_features_total: int,
    n_features_selected: int,
    acc_ga: Optional[float],
    acc_pca: Optional[float],
    acc_skb: Optional[float],
    selected_indices: Optional[List[int]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    تسجيل تجربة جديدة في قاعدة البيانات.
    """
    conn = _get_connection()
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT INTO runs (
                filename, stored_path, target_column,
                n_features_total, n_features_selected,
                acc_ga, acc_pca, acc_skb,
                selected_indices, params_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                stored_path,
                target_column,
                int(n_features_total),
                int(n_features_selected),
                float(acc_ga) if acc_ga is not None else None,
                float(acc_pca) if acc_pca is not None else None,
                float(acc_skb) if acc_skb is not None else None,
                json.dumps(selected_indices or []),
                json.dumps(params or {}),
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"[DB ERROR] فشل تسجيل السجل: {e}")
    finally:
        conn.close()


#  استرجاع السجلات 
def get_logs(limit: Optional[int] = None) -> pd.DataFrame:
    """
    إرجاع السجلات الأحدث أولًا.
    """
    conn = _get_connection()
    query = "SELECT * FROM runs ORDER BY datetime(created_at) DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    try:
        df = pd.read_sql(query, conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def get_last_run() -> Optional[dict]:
    """
    إرجاع آخر تجربة مسجلة كقاموس.
    """
    conn = _get_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM runs ORDER BY datetime(created_at) DESC LIMIT 1")
        row = c.fetchone()
        if not row:
            return None
        colnames = [d[0] for d in c.description]
        return dict(zip(colnames, row))
    except Exception as e:
        print(f"[DB ERROR] فشل قراءة آخر تجربة: {e}")
        return None
    finally:
        conn.close()
