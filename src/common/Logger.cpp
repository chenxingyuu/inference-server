#include "common/Logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <vector>

namespace infer {

void initLogger(const std::string& log_level, const std::string& log_file) {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

    if (!log_file.empty()) {
        sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file, 50 * 1024 * 1024, 3));
    }

    auto logger = std::make_shared<spdlog::logger>("infer", sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::from_str(log_level));
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
    // Flush immediately on ERROR or above so crash logs are not lost in buffers.
    spdlog::flush_on(spdlog::level::err);
}

} // namespace infer
