.PHONY: help \
	configure-cpu configure-gpu configure-npu configure-tests \
	build build-cpu build-gpu build-npu build-tests \
	build-go swagger \
	build-web dev-web \
	run run-cpu run-gpu run-npu \
	test validate clean \
	docker-build-cpu docker-build-gpu docker-build-npu docker-build-infer-server docker-build-infer-web

PNPM ?= $(shell which pnpm 2>/dev/null || echo pnpm)
WEB_DIR := tools/infer-web

BUILD_DIR ?= build
BUILD_TYPE ?= Release
CONFIG ?= config/config.yaml
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
GO   ?= $(shell which go 2>/dev/null || ls $(HOME)/sdk/go*/bin/go 2>/dev/null | sort -V | tail -1)
SWAG ?= $(shell which swag 2>/dev/null || echo $(HOME)/go/bin/swag)
GODIR := $(dir $(GO))

INFER_BIN := ./$(BUILD_DIR)/infer_server

DOCKER_REGISTRY ?= registry.cn-hangzhou.aliyuncs.com/daxx
DOCKER_PLATFORMS ?= linux/amd64,linux/arm64

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

run-cpu run-gpu run-npu:
	$(INFER_BIN) config/config.yaml

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

build-web:
	cd $(WEB_DIR) && $(PNPM) install && $(PNPM) run build

dev-web:
	cd $(WEB_DIR) && $(PNPM) install && $(PNPM) run dev

docker-build-infer-server:
	docker buildx build \
	  --platform=$(DOCKER_PLATFORMS) \
	  -t $(DOCKER_REGISTRY)/infer-server:latest \
	  -f docker/Dockerfile.infer-server \
	  --provenance=false \
	  --sbom=false \
	  --push \
	  .
docker-build-infer-web:
	docker buildx build \
	  --platform=$(DOCKER_PLATFORMS) \
	  -t $(DOCKER_REGISTRY)/infer-web:latest \
	  -f docker/Dockerfile.infer-web \
	  --provenance=false \
	  --sbom=false \
	  --push \
	  .
docker-build-cpu:
	DOCKER_BUILDKIT=1 docker build -t inference-server:cpu -f docker/Dockerfile.cpu .
	docker tag inference-server:cpu registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:cpu
	docker push registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:cpu
docker-build-gpu:
	DOCKER_BUILDKIT=1 docker build \
	  --platform=linux/amd64 \
	  -t inference-server:tensorrt \
	  -f docker/Dockerfile.tensorrt \
	  --build-arg TRT_DEVEL_IMAGE=nvcr.io/nvidia/tensorrt:24.02-py3 \
	  --build-arg TRT_RUNTIME_IMAGE=nvcr.io/nvidia/cuda:12.3.2-runtime-ubuntu22.04 \
	  .
	docker tag inference-server:tensorrt registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:tensorrt
	docker push registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:tensorrt
docker-build-npu:
	DOCKER_BUILDKIT=1 docker build \
	  -t inference-server:ascend-cann6 \
	  -f docker/Dockerfile.ascend.cann6 \
	  --build-arg ASCEND_DEVEL_IMAGE=registry.cn-hangzhou.aliyuncs.com/daxx/cann:6.0.1-310p-ubuntu20.04-py3.9 \
	  --build-arg ASCEND_RUNTIME_IMAGE=registry.cn-hangzhou.aliyuncs.com/daxx/cann:6.0.1-310p-ubuntu20.04-py3.9-runtime \
	  .
	docker tag inference-server:ascend-cann6 registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:ascend-cann6
	docker push registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:ascend-cann6