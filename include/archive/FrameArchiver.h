#pragma once

#include "common/Config.h"
#include "common/Types.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace infer {

struct FrameArchiveResult {
    std::string local_path;
    std::string upload_state{"disabled"};
};

class FrameArchiver {
public:
    explicit FrameArchiver(FrameArchiveConfig cfg);
    ~FrameArchiver();

    FrameArchiver(const FrameArchiver&) = delete;
    FrameArchiver& operator=(const FrameArchiver&) = delete;

    bool enabled() const noexcept { return cfg_.enabled; }

    // Non-blocking submit. Returns metadata for immediate Kafka publish.
    FrameArchiveResult submit(const StreamMeta& meta, const cv::Mat* frame);

private:
    struct ArchiveTask {
        std::string local_path;
        std::string object_key;
        cv::Mat     frame;
    };

    std::string buildObjectKey(const StreamMeta& meta) const;
    std::string buildLocalPath(const StreamMeta& meta) const;
    bool enqueue(ArchiveTask task);
    void workerLoop();

    FrameArchiveConfig      cfg_;
    std::atomic<bool>       stop_{false};
    std::mutex              mu_;
    std::condition_variable cv_;
    std::queue<ArchiveTask> queue_;
    std::vector<std::thread> workers_;
};

} // namespace infer
