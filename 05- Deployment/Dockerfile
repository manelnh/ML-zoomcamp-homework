FROM agrigorev/zoomcamp-model:2025

WORKDIR /app
# Install curl if not present, then install Poetry
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

COPY pyproject.toml .

# Install dependencies with Poetry (no root package)
RUN poetry config virtualenvs.create false && poetry install --no-root

COPY ["predict_lead.py", "pipeline_v1.bin", "./"]

COPY predict_lead.py  pipeline_v1.bin ./

EXPOSE 9696

CMD ["uvicorn", "predict_lead:app", "--host" ,"0.0.0.0", "--port" ,"9696"]

