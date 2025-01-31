// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>
#include <string>
#include <string_view>

namespace sleipnir {

/**
 * Records the number of profiler measurements (start/stop pairs) and the
 * average duration between each start and stop call.
 */
class SetupProfiler {
 public:
  std::string name;

  /**
   * Constructs a SetupProfiler.
   *
   * @param name Name of measurement to show in diagnostics.
   */
  explicit SetupProfiler(std::string_view name) : name{name} {}

  /**
   * Tell the profiler to start measuring setup time.
   */
  void Start() {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    m_setupStartTime = std::chrono::steady_clock::now();
#endif
  }

  /**
   * Tell the profiler to stop measuring setup time.
   */
  void Stop() {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    m_setupStopTime = std::chrono::steady_clock::now();
    m_setupDuration = m_setupStopTime - m_setupStartTime;
#endif
  }

  /**
   * The setup duration in milliseconds as a double.
   */
  const std::chrono::duration<double>& Duration() const {
    return m_setupDuration;
  }

 private:
  std::chrono::steady_clock::time_point m_setupStartTime;
  std::chrono::steady_clock::time_point m_setupStopTime;
  std::chrono::duration<double> m_setupDuration{0.0};
};

}  // namespace sleipnir
