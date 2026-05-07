.PHONY: help \
	configure-cpu configure-gpu configure-npu configure-tests \
	build build-cpu build-gpu build-npu build-tests \
	build-go swagger \
	run run-cpu run-gpu run-npu \
	test validate clean \
	docker-build-cpu docker-build-gpu docker-build-npu docker-build-infer-server \
	up up-cpu up-gpu up-npu \
	down down-cpu down-gpu down-npu

BUILD_DIR ?= build
BUILD_TYPE ?= Release
CONFIG ?= config/config.cpu.yaml
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
GO   ?= $(shell which go 2>/dev/null || ls $(HOME)/sdk/go*/bin/go 2>/dev/null | sort -V | tail -1)
SWAG ?= $(shell which swag 2>/dev/null || echo $(HOME)/go/bin/swag)
GODIR := $(dir $(GO))

INFER_BIN := ./$(BUILD_DIR)/infer_server

help:
	@echo "Common targets:"
	@echo "  make build              # default: CPU build"
	@echo "  make build-cpu          # ONNX Runtime (CPU/MPS)"
	@echo "  make build-gpu          # TensorRT build"
	@echo "  make build-npu          # Ascend build"
	@echo "  make build-tests        # build with tests enabled"
	@echo "  make build-go           # build Go tools (infer-ctl, infer-server) to tools/bin/"
	@echo "  make run                # default: run with $(CONFIG)"
	@echo "  make run CONFIG=...     # override config path"
	@echo "  make test               # run ctest"
	@echo "  make validate           # run scripts/validate-repo.sh"
	@echo "  make clean              # remove build dir (recommended before backend switch)"
	@echo "  make up|up-cpu|up-gpu|up-npu"
	@echo "  make down|down-cpu|down-gpu|down-npu"
	@echo "  make docker-build-cpu|docker-build-gpu|docker-build-npu|docker-build-infer-server"

configure-cpu:
	cmake -B $(BUILD_DIR) \
	  -DBUILD_TRT_BACKEND=OFF \
	  -DBUILD_ASCEND_BACKEND=OFF \
	  -DBUILD_ONNX_BACKEND=ON \
	  -DBUILD_ONNX_BACKEND_COREML=ON \
	  -DBUILD_REDIS_PUBLISHER=ON \
	  -DBUILD_GRPC_PUBLISHER=ON \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

configure-gpu:
	cmake -B $(BUILD_DIR) \
	  -DBUILD_TRT_BACKEND=ON \
	  -DBUILD_ASCEND_BACKEND=OFF \
	  -DBUILD_ONNX_BACKEND=OFF \
	  -DBUILD_REDIS_PUBLISHER=ON \
	  -DBUILD_GRPC_PUBLISHER=ON \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

configure-npu:
	cmake -B $(BUILD_DIR) \
	  -DBUILD_TRT_BACKEND=OFF \
	  -DBUILD_ASCEND_BACKEND=ON \
	  -DBUILD_ONNX_BACKEND=OFF \
	  -DBUILD_REDIS_PUBLISHER=ON \
	  -DBUILD_GRPC_PUBLISHER=ON \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

configure-tests:
	cmake -B $(BUILD_DIR) \
	  -DBUILD_TRT_BACKEND=OFF \
	  -DBUILD_ASCEND_BACKEND=OFF \
	  -DBUILD_ONNX_BACKEND=ON \
	  -DBUILD_TESTS=ON \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: build-cpu

build-cpu: configure-cpu
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

build-gpu: configure-gpu
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

build-npu: configure-npu
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

build-tests: configure-tests
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

run:
	$(INFER_BIN) "$(CONFIG)"

run-cpu:
	$(INFER_BIN) config/config.cpu.yaml

run-gpu:
	$(INFER_BIN) config/config.gpu.yaml

run-npu:
	$(INFER_BIN) config/config.npu.yaml

test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

validate:
	bash scripts/validate-repo.sh

clean:
	rm -rf $(BUILD_DIR)

swagger:
	cd tools && PATH="$(GODIR):$$PATH" $(SWAG) init -g infer-server/main.go -o infer-server/docs --parseDependency

build-go: swagger
	cd tools && $(GO) build -o bin/infer-ctl ./infer-ctl/
	cd tools && $(GO) build -o bin/infer-server ./infer-server/

docker-build-infer-server:
	DOCKER_BUILDKIT=1 docker build -t infer-server:latest -f docker/Dockerfile.infer-server .

docker-build-cpu:
	DOCKER_BUILDKIT=1 docker build -t inference-server:cpu -f docker/Dockerfile.cpu .

docker-build-gpu:
	DOCKER_BUILDKIT=1 docker build \
	  -t inference-server:tensorrt \
	  -f docker/Dockerfile.tensorrt \
	  --build-arg TRT_DEVEL_IMAGE=nvcr.io/nvidia/tensorrt:24.02-py3 \
	  --build-arg TRT_RUNTIME_IMAGE=nvcr.io/nvidia/tensorrt:24.02-py3 \
	  .

docker-build-npu:
	DOCKER_BUILDKIT=1 docker build \
	  -t inference-server:ascend-cann6 \
	  -f docker/Dockerfile.ascend.cann6 \
	  --build-arg ASCEND_BASE_IMAGE=ascendai/cann:6.0.1-310p-ubuntu20.04-py3.9 \
	  .

up: up-cpu

up-cpu:
	docker compose -f docker/docker-compose.cpu.yml up -d

up-gpu:
	docker compose -f docker/docker-compose.nvidia.yml up -d

up-npu:
	docker compose -f docker/docker-compose.ascend.yml up -d

down-cpu:
	docker compose -f docker/docker-compose.cpu.yml down

down: down-cpu

down-gpu:
	docker compose -f docker/docker-compose.nvidia.yml down

down-npu:
	docker compose -f docker/docker-compose.ascend.yml down
