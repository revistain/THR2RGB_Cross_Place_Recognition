USER_UID := $(shell id -u)
USER_GID := $(shell id -g)
USER_NAME := $(shell whoami)

# 이미지 및 사용자 정보 변수
IMAGE_NAME := ${USER_NAME}/thr2rgb_pr
IMAGE_TAG := cuda12.1.0-torch2.5.1
# FULL_IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)
FULL_IMAGE := sjkwon/thr2rgb_pr:cuda12.1.0-torch2.5.1   
 

CONTAINER_NAME := jwkim_thr2rgb_pr

POS_GPUS=all
TBP=8818
JNBP=8819

docker-build:
	docker build --build-arg USERNAME=${USER_NAME} \
	             --build-arg USER_UID=$(USER_UID) \
	             --build-arg USER_GID=$(USER_GID) \
				 -f Dockerfile \
	             -t $(FULL_IMAGE) .

docker-run:
	docker run --gpus all -it \
		--name $(CONTAINER_NAME) \
		--ipc=host \
		-p ${TBP}:6006 \
		-p ${JNBP}:8888 \
		-e "TERM=xterm-256color" \
		-w /home/${USER_NAME}/workspace \
		-v /home/${USER_NAME}/workspace:/home/${USER_NAME}/workspace \
		-v /data:/data \
		-v /DATA1:/DATA1 \
		-v /DATA2:/DATA2 \
		-v /DATA3:/DATA3 \
		-v /DATA4:/DATA4 \
		-v /raid:/raid \
		-v /raid2:/raid2: \
		-v /raid3:/raid3: \
		-v /raid4:/raid4: \
		-v /raid5:/raid5: \
		-v /raid6:/raid6: \
		-v /home/sjkwon:/home/sjkwon: \
		-e NVIDIA_VISIBLE_DEVICES=${POS_GPUS} \
		--shm-size=32G \
		$(FULL_IMAGE) \
		/bin/bash
