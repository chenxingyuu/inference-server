#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>

namespace infer {

struct SourceEventTime {
    std::chrono::steady_clock::time_point steady;
    int64_t epoch_ms;

    static SourceEventTime now() {
        using namespace std::chrono;
        return {steady_clock::now(),
                duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()};
    }
};

class SourceObservation {
public:
    enum class Phase { Idle, Connecting, Reconnecting, Open };

    struct Snapshot {
        Phase phase{Phase::Idle};
        SourceEventTime changed{};
        std::optional<SourceEventTime> last_frame;
        std::optional<SourceEventTime> reconnecting_since;
    };

    void transition(Phase phase) {
        std::lock_guard lock(mu_);
        if (state_.phase == phase) return;
        const auto now = SourceEventTime::now();
        std::optional<SourceEventTime> reconnecting_since;
        if (phase == Phase::Reconnecting) reconnecting_since = now;
        else if (phase == Phase::Open) reconnecting_since = state_.reconnecting_since;
        state_ = {phase, now, {}, reconnecting_since};
    }

    void decoded() {
        std::lock_guard lock(mu_);
        if (state_.phase == Phase::Open) {
            state_.last_frame = SourceEventTime::now();
            state_.reconnecting_since.reset();
        }
    }

    Snapshot snapshot() const {
        std::lock_guard lock(mu_);
        return state_;
    }

private:
    mutable std::mutex mu_;
    Snapshot state_;
};

} // namespace infer
