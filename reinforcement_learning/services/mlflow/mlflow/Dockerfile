FROM python:3.7

RUN addgroup --system --gid 1000 stranger && \
     adduser --system --uid 1000 stranger --ingroup stranger

USER stranger

# Install python packages
RUN pip install mlflow boto3 psycopg2-binary pysftp
RUN pip install paramiko==2.7.2
