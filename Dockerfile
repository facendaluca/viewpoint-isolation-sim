FROM python:3.12-slim

# curl is needed for the HEALTHCHECK CMD below
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for runtime security
RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid 1001 --no-create-home --no-log-init --shell /bin/false appuser

# Streamlit 1.10.0+ requires a non-root WORKDIR
# This also matches REPO_ROOT derivation in ui/app_state.py (__file__-anchored to /app)
WORKDIR /app

# --- Install phase (cache-friendly) ---
# Invalidated only when pyproject.toml or src/ change, not when pages/ui/configs change
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# --- Application files ---
COPY Dashboard.py ./
COPY pages/ ./pages/
COPY ui/ ./ui/
COPY configs/ ./configs/

# Pre-create outputs dir owned by appuser so the named volume initialises with correct
# ownership on first mount (Docker copies this directory's contents into a new volume)
RUN mkdir -p outputs/runs \
    && chown -R appuser:appuser outputs/

USER appuser

EXPOSE 8501

# Streamlit's own health endpoint — available since 1.10.0
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Exec form: signals pass directly to streamlit, not via a shell wrapper
ENTRYPOINT ["streamlit", "run", "Dashboard.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
