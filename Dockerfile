FROM python:3.10.13-slim
WORKDIR /app
COPY main.py /app
COPY constants.py /app
ADD models /app/models
COPY requirements.txt /app
RUN pip install -r requirements.txt
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000" ]
