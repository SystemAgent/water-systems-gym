FROM python:3.7

RUN addgroup --system --gid 1000 stageai && \
     adduser --system --uid 1000 stageai --ingroup stageai

USER stageai

# Install python packages
RUN pip install mlflow boto3 psycopg2-binary pysftp
RUN pip install paramiko==2.7.2
