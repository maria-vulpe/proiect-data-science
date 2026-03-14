# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System libs needed to compile / run native extensions
# Note: libgl1-mesa-glx was replaced by libgl1 in Debian Bookworm (python:3.12-slim base)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Core data / ML stack ─────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.3 \
        scipy==1.13.1 \
        scikit-learn==1.5.2 \
        xgboost==2.1.3 \
        catboost==1.2.7

# ── PyTorch (CPU-only — keeps the image ~2 GB smaller than CUDA) ─────────────
RUN pip install --no-cache-dir \
        torch==2.4.1 \
        torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Computer vision & medical imaging ───────────────────────────────────────
RUN pip install --no-cache-dir \
        opencv-python-headless==4.10.0.84 \
        Pillow==10.4.0 \
        nibabel==5.3.2

# ── Visualisation ────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        matplotlib==3.9.4 \
        seaborn==0.13.2 \
        altair==5.4.1 \
        plotly==5.24.1

# ── Streamlit & UI ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        streamlit==1.39.0 \
        streamlit-option-menu==0.3.13

# ── PDF generation ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        reportlab==4.2.5

# ── Backend / cloud ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        python-dotenv==1.0.1 \
        supabase==2.9.1


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Runtime system libs (no build tools needed here)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source code
COPY main.py              ./
COPY home.py              ./
COPY clinician.py         ./
COPY data_scientist.py    ./
COPY data_scientist_models.py ./
COPY cardiac_angiography.py   ./
COPY heart_reconstruction_diagnosis.py ./
COPY model_utils.py       ./
COPY prepare_external_datasets.py ./
COPY data_cleaning.py     ./
COPY utils/               ./utils/

# Copy datasets and pre-trained model weights
COPY data/                ./data/
COPY data-gender/         ./data-gender/
COPY models_seg_dig/      ./models_seg_dig/

# ── Streamlit configuration ───────────────────────────────────────────────────
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n' \
    > /root/.streamlit/config.toml

# Expose Streamlit default port
EXPOSE 8501

# ── Environment variables ─────────────────────────────────────────────────────
# Pass secrets at runtime — never bake them into the image:
#   docker run -p 8501:8501 --env-file .env healthtrack-ai
# or: docker run -e SUPABASE_URL=... -e SUPABASE_SERVICE_KEY=... ...
ENV DATA_DIR="./data"

# Health-check (polls the Streamlit health endpoint every 30 s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the app
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
