FROM agrigorev/zoomcamp-model:2025

WORKDIR /homework

COPY predict_lead.py predict_lead_test.py pyproject.toml pipeline_v1.bin /homework/

RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN poetry config virtualenvs.create false && poetry install --no-root

EXPOSE 9697

CMD ["uvicorn", "predict_lead:app", "--host", "0.0.0.0", "--port", "9697"]

#must run docker desktop
#build docker container: docker build -t zoomcamp-fastapi .
#use docker run zoomcamp-fastapi (or docker run -p 9697:9697 zoomcamp-fastapi) to run the project in powershell

#output using response command:  $client = @{
#    lead_source = "organic_search"
#    number_of_courses_viewed = 4
#   annual_income = 80304.0 }

#output calling predict_lead_test.py file that exists in the container : 
#to check wether files are is container or not:
#docker exec -it container_name /bin/sh

#docker exec -it container_name python3 /homework/predict_lead_test.py
