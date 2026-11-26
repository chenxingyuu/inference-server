#include "server/ModelProbe.h"
#include "server/ModelTest.h"

#include "common/RuntimeState.h"

#include "common/Config.h"
#include "infer/BackendFactory.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <optional>
#include <fstream>
#include <set>
#include <spawn.h>
#include <stdexcept>
#include <system_error>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <opencv2/imgcodecs.hpp>

extern "C" {
#include <libavutil/base64.h>
}

extern char** environ;

namespace infer {
namespace {

constexpr auto kProbeTimeout = std::chrono::seconds(90);
constexpr char kProbeErrorFile[] = ".probe-error";

} // namespace

int runModelProbe(const std::filesystem::path& model_dir) {
    try {
        auto model = loadRepositoryModel(model_dir);
        auto checkDevice = [&](int device) {
            model.device_id = device;
            try {
                auto backend = createBackend(model);
                backend->loadModel(model);
                if (!backend->isLoaded()) throw std::runtime_error("backend did not load the weights");
                backend->unloadModel();
            } catch (const std::exception& e) {
                throw std::runtime_error("device " + std::to_string(device) + ": " + e.what());
            }
        };
        if (model.device_ids.empty()) {
            checkDevice(model.device_id);
        } else {
            const std::set<int> devices(model.device_ids.begin(), model.device_ids.end());
            for (int device : devices) checkDevice(device);
        }
        return 0;
    } catch (const std::exception& e) {
        std::ofstream(model_dir / kProbeErrorFile) << std::string(e.what()).substr(0, 1024) << '\n';
        return 1;
    }
}

namespace {

std::optional<int> runChild(const std::string& executable, const char* flag,
                            const std::filesystem::path& dir, char** environment) {
    auto program = executable;
    std::string option = flag;
    auto path = dir.string();
    char* args[] = {program.data(), option.data(), path.data(), nullptr};
    pid_t pid{};
    if (const int rc = posix_spawnp(&pid, program.c_str(), nullptr, nullptr, args, environment); rc != 0) {
        throw std::system_error(rc, std::generic_category(), "start model check");
    }

    const auto deadline = std::chrono::steady_clock::now() + kProbeTimeout;
    int status{};
    while (true) {
        const auto result = waitpid(pid, &status, WNOHANG);
        if (result == pid) return status;
        if (result < 0 && errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "wait for model check");
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            return std::nullopt;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

nlohmann::json testError(const char* code, const char* message) {
    return {{"status", "error"}, {"code", code}, {"message", message}};
}

class TestDirectory {
public:
    TestDirectory() {
        auto path_template = (std::filesystem::temp_directory_path() / "infer-model-test.XXXXXX").string();
        if (!mkdtemp(path_template.data()))
            throw std::system_error(errno, std::generic_category(), "create model test directory");
        path = std::move(path_template);
    }
    ~TestDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    TestDirectory(const TestDirectory&) = delete;
    TestDirectory& operator=(const TestDirectory&) = delete;

    std::filesystem::path path;
};

bool isSupportedImage(const std::vector<unsigned char>& image) {
    return (image.size() >= 3 && image[0] == 0xff && image[1] == 0xd8 && image[2] == 0xff) ||
           (image.size() >= 8 && image[0] == 0x89 && image[1] == 'P' && image[2] == 'N' && image[3] == 'G' &&
            image[4] == 0x0d && image[5] == 0x0a && image[6] == 0x1a && image[7] == 0x0a);
}

} // namespace

void probeModelInChild(const std::string& executable, const std::filesystem::path& model_dir) {
    const auto status = runChild(executable, "--probe-model", model_dir, environ);
    if (!status) throw std::runtime_error("weight loading timed out");
    if (WIFEXITED(*status) && WEXITSTATUS(*status) == 0) return;
    std::ifstream error(model_dir / kProbeErrorFile);
    std::string message;
    std::getline(error, message);
    throw std::runtime_error(message.empty() ? "weight loading process exited unexpectedly" : message);
}

int runModelTest(const std::filesystem::path& request_dir) {
    auto test = [&]() {
        cv::Mat image;
        try {
            image = cv::imread((request_dir / "image").string(), cv::IMREAD_COLOR);
        } catch (const cv::Exception&) {
            return testError("invalid_image", "could not decode the image");
        }
        if (image.empty() || image.cols > kModelTestMaxImageSide || image.rows > kModelTestMaxImageSide)
            return testError("invalid_image", "could not decode the image within the supported dimensions");

        auto snapshot = loadRuntimeState((request_dir / "model.json").string());
        if (snapshot.added_models.size() != 1)
            throw std::runtime_error("model test configuration is missing");

        auto& model = snapshot.added_models.front();
        if (!model.device_ids.empty()) model.device_id = model.device_ids.front();
        model.buffer_pool_size = 1;
        if (model.backend == DeviceType::CUDA) model.batch_size = 1;
        if (model.backend == DeviceType::Ascend && !model.om_paths.empty())
            model.om_paths = {*model.om_paths.begin()};

        auto backend = createBackend(model);
        backend->loadModel(model);
        if (!backend->isLoaded()) throw std::runtime_error("model weights did not load");
        return nlohmann::json{{"status", "ok"}, {"data", evaluateModelImage(model, *backend, image)}};
    };

    nlohmann::json result;
    try {
        result = test();
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "Model test: %s\n", error.what());
        result = testError("model_test_unsupported_model", "model input or output is incompatible with single-image testing");
    } catch (const std::exception& error) {
        // Keep filesystem paths and backend diagnostics in engine logs.
        std::fprintf(stderr, "Model test: %s\n", error.what());
        result = testError("model_test_failed", "could not run single-image inference with this model");
    }

    auto payload = result.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    if (payload.size() > kModelTestMaxResponseBytes)
        payload = testError("model_test_result_too_large", "the model produced a result that is too large").dump();
    std::ofstream output(request_dir / "result.json", std::ios::binary);
    output << payload;
    output.close();
    return output ? 0 : 1;
}

nlohmann::json testModelInChild(const std::string& executable, const ModelConfig& model,
                               const std::string& encoded_image) {
    if (encoded_image.size() > AV_BASE64_SIZE(kModelTestMaxImageBytes) - 1)
        return testError("image_too_large", "image exceeds 8 MiB");
    if (encoded_image.empty() || encoded_image.find('\0') != std::string::npos)
        return testError("invalid_image", "invalid encoded image");

    std::vector<unsigned char> image(encoded_image.size() / 4 * 3 + 3);
    const int image_size = av_base64_decode(image.data(), encoded_image.c_str(), image.size());
    if (image_size <= 0) return testError("invalid_image", "invalid encoded image");
    if (static_cast<size_t>(image_size) > kModelTestMaxImageBytes)
        return testError("image_too_large", "image exceeds 8 MiB");
    image.resize(image_size);
    if (!isSupportedImage(image)) return testError("invalid_image", "upload one JPEG or PNG image");

    try {
        const TestDirectory request;
        RuntimeState snapshot;
        snapshot.added_models.push_back(model);
        const auto model_path = request.path / "model.json";
        saveRuntimeState(snapshot, model_path.string());
        if (!std::filesystem::is_regular_file(model_path))
            throw std::runtime_error("could not save test configuration");
        std::ofstream file(request.path / "image", std::ios::binary);
        file.write(reinterpret_cast<const char*>(image.data()), image.size());
        file.close();
        if (!file) throw std::runtime_error("could not write the test image");

        const std::vector<std::string> limits = {
            "OPENCV_IO_MAX_IMAGE_WIDTH", "OPENCV_IO_MAX_IMAGE_HEIGHT", "OPENCV_IO_MAX_IMAGE_PIXELS"
        };
        std::vector<std::string> environment;
        for (char** value = environ; *value; ++value) {
            const std::string_view entry(*value);
            if (std::none_of(limits.begin(), limits.end(), [&](const auto& key) {
                    return entry.substr(0, entry.find('=')) == key;
                })) environment.emplace_back(entry);
        }
        const auto max_side = std::to_string(kModelTestMaxImageSide);
        environment.push_back(limits[0] + "=" + max_side);
        environment.push_back(limits[1] + "=" + max_side);
        environment.push_back(limits[2] + "=" + std::to_string(kModelTestMaxImageSide * kModelTestMaxImageSide));
        std::vector<char*> entries;
        for (auto& entry : environment) entries.push_back(entry.data());
        entries.push_back(nullptr);
        const auto status = runChild(executable, "--test-model", request.path, entries.data());
        if (!status) return testError("model_test_timeout", "model test timed out");
        if (!WIFEXITED(*status) || WEXITSTATUS(*status) != 0)
            return testError("model_test_failed", "the model test process exited unexpectedly");

        const auto result_path = request.path / "result.json";
        if (std::filesystem::file_size(result_path) > kModelTestMaxResponseBytes)
            return testError("model_test_result_too_large", "the model produced a result that is too large");
        std::ifstream result(result_path, std::ios::binary);
        return nlohmann::json::parse(result);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Model test: %s\n", error.what());
        return testError("model_test_unavailable", "could not complete the model test");
    }
}

} // namespace infer
