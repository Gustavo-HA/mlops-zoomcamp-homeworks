FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install pipenv

COPY ["score.py", "./"]
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

RUN pip list

CMD [ "python", "score.py", "yellow", "2023", "5" ]