#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <optional>
#include <spawn.h>
#include <sys/wait.h>
#include <system_error>
#include <thread>

namespace infer {

inline std::optional<int> runBoundedChild(
    char* const argv[], char* const environment[], std::chrono::milliseconds timeout,
    const posix_spawn_file_actions_t* actions = nullptr) {
    pid_t pid{};
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    if (const int rc = posix_spawnp(&pid, argv[0], actions, nullptr, argv, environment); rc != 0) {
        throw std::system_error(rc, std::generic_category(), "start child process");
    }

    int status{};
    while (true) {
        const auto result = waitpid(pid, &status, WNOHANG);
        if (result == pid) return status;
        if (result < 0 && errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "wait for child process");
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            return std::nullopt;
        }
        std::this_thread::sleep_until(std::min(deadline,
            std::chrono::steady_clock::now() + std::chrono::milliseconds(25)));
    }
}

} // namespace infer
