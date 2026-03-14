
from dotenv import load_dotenv
import os, uuid, datetime as dt, math
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError(
        "FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY not set in environment!"
    )

print("➤ Supabase URL:", SUPABASE_URL)
print("➤ Supabase Key starts with:", SUPABASE_SERVICE_KEY[:10], "...")

SUPA: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
BUCKET     = SUPA.storage.from_("models")

def _get_allowed_columns():

    return {
        "model_type",
        "dataset",
        "auc",
        "git_sha",
        "type",
    }

def push_artifact(local_path: str, meta: dict) -> str:

    ts   = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    ext  = os.path.splitext(local_path)[1]          # ex: ".pkl"
    dest = f"{ts}_{uuid.uuid4().hex[:4]}{ext}"       # ex: "2025-06-02T14-30-00_a3f1.pkl"

    try:
        with open(local_path, "rb") as f:
            BUCKET.upload(dest, f, file_options={"content-type": "application/octet-stream"})
    except Exception as e:
        err_str = str(e).lower()
        if "413" in err_str or "payload too large" in err_str:
            raise RuntimeError(
                f"File too large for cloud upload (413). Model saved locally at '{local_path}'."
            )
        raise

    allowed    = _get_allowed_columns()
    clean_meta = {}
    for k, v in meta.items():
        if k not in allowed:
            continue
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        clean_meta[k] = v

    SUPA.table("model_registry").insert({
        **clean_meta,
        "filename": dest
    }).execute()

    return dest

def list_artifacts(limit: int = 30) -> list[dict]:

    resp = (
        SUPA
        .table("model_registry")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data

def fetch_bytes(filename: str) -> bytes:

    data = BUCKET.download(filename)
    if isinstance(data, bytes):
        return data
    return data.read()
