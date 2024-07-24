FROM --platform=linux/amd64 docker.io/library/python:3.11-slim

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user data /opt/app/data

# Install Python dependencies in requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Copy the package files
COPY --chown=user:user evaluate.py /opt/app/
COPY --chown=user:user src /opt/app/src
COPY --chown=user:user setup.py /opt/app/

# Install the package
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    /opt/app

# Install opencv 
RUN python -m pip install --user opencv_python
RUN python -m pip install --user opencv-python-headless

ENV GRAND_CHALLENGE_MAX_WORKERS=4

# Run the evaluation script
ENTRYPOINT ["python", "evaluate.py"]
