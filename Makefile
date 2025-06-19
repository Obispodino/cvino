#======================#
#     Configuration    #
#======================#

GCP_PROJECT := wagon-bootcamp-457511
GCP_REGION := europe-west1
DOCKER_REPO_NAME := cvino-repo
DOCKER_IMAGE_NAME := cvino-api
DOCKER_LOCAL_PORT := 8080
GAR_MEMORY := 2Gi

DOCKER_IMAGE_PATH := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME)

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
#          GCP         #
#======================#

gcloud-set-project:
	gcloud config set project $(GCP_PROJECT)

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

docker_show_image_path:
	@echo $(DOCKER_IMAGE_PATH)

docker_build:
	docker build \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

docker_build_alternative:
	docker buildx build --load \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

docker_tag_local:
	docker tag $(DOCKER_IMAGE_NAME):local $(DOCKER_IMAGE_PATH):prod

docker_run:
	docker run \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_PATH):prod

docker_run_interactively:
	docker run -it \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_PATH):prod \
		bash

#======================#
#  Docker + Deployment #
#======================#

docker_allow:
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev

docker_create_repo:
	gcloud artifacts repositories create $(DOCKER_REPO_NAME) \
		--repository-format=docker \
		--location=$(GCP_REGION) \
		--description="Repository for storing docker images"

docker_push:
	docker push $(DOCKER_IMAGE_PATH):prod

docker_deploy:
	gcloud run deploy \
		--image $(DOCKER_IMAGE_PATH):prod \
		--memory $(GAR_MEMORY) \
		--region $(GCP_REGION) \
		--env-vars-file .env.yaml
