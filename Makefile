style_check:
	ruff check text2sql_training/

style_fix:
	ruff format text2sql_training/

docker_build:
	docker build -t fine-tune-llm-in-2024:latest -f Dockerfile .

docker_run:
	docker run -it --gpus all --ipc=host --net=host -v $PWD:/app fine-tune-llm-in-2024:latest /bin/bash