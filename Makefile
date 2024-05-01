TOP-K?=10

run:
	python3 main.py --dataset-name ${DATASET-NAME} --top-k ${TOP-K}

format:
	python3 formater.py --dataset-name ${DATASET-NAME}
