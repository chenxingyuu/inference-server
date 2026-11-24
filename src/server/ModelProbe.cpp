#include "server/ModelProbe.h"

#include "common/Config.h"
#include "infer/BackendFactory.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <fstream>
#include <set>
#include <spawn.h>
#include <stdexcept>
#include <system_error>
#include <sys/wait.h>
#include <thread>

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

void probeModelInChild(const std::string& executable, const std::filesystem::path& model_dir) {
    auto program = executable;
    std::string option = "--probe-model";
    auto path = model_dir.string();
    char* args[] = {program.data(), option.data(), path.data(), nullptr};
    pid_t pid{};
    if (const int rc = posix_spawnp(&pid, program.c_str(), nullptr, nullptr, args, environ); rc != 0) {
        throw std::system_error(rc, std::generic_category(), "start model check");
    }

    const auto deadline = std::chrono::steady_clock::now() + kProbeTimeout;
    int status{};
    while (true) {
        const auto result = waitpid(pid, &status, WNOHANG);
        if (result == pid) break;
        if (result < 0 && errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "wait for model check");
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            throw std::runtime_error("weight loading timed out");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return;
    std::ifstream error(model_dir / kProbeErrorFile);
    std::string message;
    std::getline(error, message);
    throw std::runtime_error(message.empty() ? "weight loading process exited unexpectedly" : message);
}

} // namespace infer
