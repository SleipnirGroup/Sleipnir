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
class SolveProfiler {
 public:
  std::string name;

  /**
   * Constructs a SolveProfiler.
   */
  SolveProfiler() = default;

  /**
   * Constructs a SolveProfiler.
   *
   * @param name Name of measurement to show in diagnostics.
   */
  explicit SolveProfiler(std::string_view name) : name{name} {}

  /**
   * Tell the profiler to start measuring solve time.
   */
  void Start() {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    m_currentSolveStartTime = std::chrono::steady_clock::now();
#endif
  }

  /**
   * Tell the profiler to stop measuring solve time, increment the number of
   * averages, and incorporate the latest measurement into the average.
   */
  void Stop() {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    m_currentSolveStopTime = std::chrono::steady_clock::now();
    m_currentSolveDuration = m_currentSolveStopTime - m_currentSolveStartTime;
    m_totalSolveDuration += m_currentSolveDuration;

    ++m_numSolves;
    m_averageSolveDuration =
        (m_numSolves - 1.0) / m_numSolves * m_averageSolveDuration +
        1.0 / m_numSolves * m_currentSolveDuration;
#endif
  }

  /**
   * The number of solves.
   */
  int NumSolves() const { return m_numSolves; }

  /**
   * The most recent solve duration in milliseconds as a double.
   */
  const std::chrono::duration<double>& CurrentDuration() const {
    return m_currentSolveDuration;
  }

  /**
   * The average solve duration in milliseconds as a double.
   */
  const std::chrono::duration<double>& AverageDuration() const {
    return m_averageSolveDuration;
  }

  /**
   * The sum of all solve durations in milliseconds as a double.
   */
  const std::chrono::duration<double>& TotalDuration() const {
    return m_totalSolveDuration;
  }

 private:
  std::chrono::steady_clock::time_point m_currentSolveStartTime;
  std::chrono::steady_clock::time_point m_currentSolveStopTime;
  std::chrono::duration<double> m_currentSolveDuration{0.0};
  std::chrono::duration<double> m_totalSolveDuration{0.0};

  int m_numSolves = 0;
  std::chrono::duration<double> m_averageSolveDuration{0.0};
};

}  // namespace sleipnir
