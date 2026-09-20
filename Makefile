#======================#
#     Configuration    #
#======================#

DOCKER_IMAGE_NAME := fast-cvino
DOCKER_LOCAL_PORT := 8000

#======================#
# Install, clean, test #
#======================#

install_requirements:
	pip install -r requirements.txt

install:
	pip install . -U

clean:
	rm -f */version.txt
	rm -f .coverage
	rm -fr */__pycache__ */*.pyc __pycache__
	rm -fr build dist
	rm -fr proj-*.dist-info
	rm -fr proj.egg-info

test_structure:
	bash tests/test_structure.sh

#======================#
#          API         #
#======================#

run_api:
	uvicorn API.fast:app --reload --port 8000

#======================#
#         Docker       #
#======================#

docker_build_local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local .

docker_run_local:
	docker run \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local

docker_run_local_interactively:
	docker run -it \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local \
		bash
