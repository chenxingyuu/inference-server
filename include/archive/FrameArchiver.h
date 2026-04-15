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

namespace infer {

struct FrameArchiveResult {
    std::string local_path;
    std::string frame_url;
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
    bool uploadToMinio(const std::string& local_path, const std::string& object_key) const;

    FrameArchiveConfig      cfg_;
    std::atomic<bool>       stop_{false};
    std::mutex              mu_;
    std::condition_variable cv_;
    std::queue<ArchiveTask> queue_;
    std::thread             worker_;
};

} // namespace infer
