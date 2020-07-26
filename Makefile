ROOT_DIR = ${PWD}

export PYTHONPATH=$(ROOT_DIR)




python:
	pipenv run python $(s)


jupyter:
	pipenv run jupyter lab