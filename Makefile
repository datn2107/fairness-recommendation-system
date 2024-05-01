TOP-K?=10
DATASET-DIR?=./datasets

run:
	python3 main.py --dataset-name ${DATASET-NAME} --dataset-dir ${DATASET-DIR} --top-k ${TOP-K}

format:
	python3 formater.py --dataset-name ${DATASET-NAME} --dataset-dir ${DATASET-DIR}
