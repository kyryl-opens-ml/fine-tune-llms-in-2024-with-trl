FROM huggingface/transformers-pytorch-gpu:4.35.2

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN MAX_JOBS=4 pip3 install flash-attn==2.5.7 --no-build-isolation

ENV DAGSTER_HOME /app/dagster_data
RUN mkdir -p $DAGSTER_HOME

ENV PYTHONPATH /app
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY text2sql_training text2sql_training 
CMD dagster dev -f text2sql_training/llm_stf.py -p 3000 -h 0.0.0.0
