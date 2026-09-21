# Build:  docker build -t tsagent-sandbox:latest docker/sandbox
#
# For reproducibility, pin the base image by digest once it works for you:
#   docker pull python:3.12-slim && docker inspect --format '{{index .RepoDigests 0}}' python:3.12-slim
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/tmp

# Pinned in requirements.txt (audited in CI). pip is removed afterwards so nothing
# can be installed at runtime.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt \
 && pip uninstall -y pip setuptools wheel

COPY runner.py /sandbox/runner.py

# Unprivileged user (nobody). The host also passes --user, belt and braces.
USER 65534:65534
WORKDIR /tmp

# -I: isolated mode (ignores PYTHON* env vars and user site-packages)
ENTRYPOINT ["python", "-I", "-B", "/sandbox/runner.py"]
