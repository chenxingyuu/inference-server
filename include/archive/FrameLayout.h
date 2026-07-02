#pragma once

#include "common/Types.h"
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace infer {
namespace frame_layout {

inline bool isSafeStreamId(const std::string& stream_id) {
    if (stream_id.empty()) return false;
    for (const char c : stream_id) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                        c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

// Portable UTC epoch from calendar fields (timegm is GNU/BSD).
inline std::time_t utcEpoch(int year, int month, int day, int hour, int minute, int second = 0) {
    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon  = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min  = minute;
    tm.tm_sec  = second;
    tm.tm_isdst = 0;
#if defined(_WIN32)
    return _mkgmtime(&tm);
#else
    return timegm(&tm);
#endif
}

inline std::optional<std::time_t> parseDayDir(const std::string& name) {
    if (name.size() != 8) return std::nullopt;
    for (char c : name) {
        if (c < '0' || c > '9') return std::nullopt;
    }
    const int year  = std::stoi(name.substr(0, 4));
    const int month = std::stoi(name.substr(4, 2));
    const int day   = std::stoi(name.substr(6, 2));
    if (month < 1 || month > 12 || day < 1 || day > 31) return std::nullopt;
    return utcEpoch(year, month, day, 0, 0, 0);
}

inline std::optional<int> parseHourDir(const std::string& name) {
    if (name.size() != 2) return std::nullopt;
    for (char c : name) {
        if (c < '0' || c > '9') return std::nullopt;
    }
    const int hour = std::stoi(name);
    if (hour < 0 || hour > 23) return std::nullopt;
    return hour;
}

inline std::optional<int> parseMinuteDir(const std::string& name) {
    if (name.size() != 2) return std::nullopt;
    for (char c : name) {
        if (c < '0' || c > '9') return std::nullopt;
    }
    const int minute = std::stoi(name);
    if (minute < 0 || minute > 59) return std::nullopt;
    return minute;
}

inline std::time_t bucketEndEpoch(int year, int month, int day, int hour, int minute) {
    return utcEpoch(year, month, day, hour, minute, 0) + 60;
}

inline std::string objectKey(const StreamMeta& meta) {
    if (!isSafeStreamId(meta.stream_id)) {
        throw std::runtime_error("FrameLayout: invalid stream_id for object key");
    }
    const auto ts = static_cast<std::time_t>(meta.capture_ts);
    std::tm tm_buf{};
#if defined(_WIN32)
    gmtime_s(&tm_buf, &ts);
#else
    gmtime_r(&ts, &tm_buf);
#endif
    const int64_t ts_ms = static_cast<int64_t>(meta.capture_ts * 1000.0);
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(4) << (tm_buf.tm_year + 1900)
        << std::setw(2) << (tm_buf.tm_mon + 1)
        << std::setw(2) << tm_buf.tm_mday
        << '/'
        << std::setw(2) << tm_buf.tm_hour
        << '/'
        << std::setw(2) << tm_buf.tm_min
        << '/'
        << meta.stream_id
        << '/'
        << ts_ms << '_' << meta.frame_seq << ".jpg";
    return oss.str();
}

} // namespace frame_layout
} // namespace infer
