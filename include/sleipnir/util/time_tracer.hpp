// Copyright (c) Sleipnir contributors

#pragma once

#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace slp {

/// Records the duration between the most recent start and stop call.
///
/// Unlike SolveTimeTracer, this doesn't count measurements or average them, so
/// it's meant for steps that run once, like autodiff setup.
class SetupTimeTracer {
 public:
  /// Constructs an enabled tracer.
  SetupTimeTracer() = default;

  /// Constructs a tracer.
  ///
  /// @param enabled Whether to measure time.
  explicit SetupTimeTracer(bool enabled) : m_enabled{enabled} {}

  /// Starts setup time measurement.
  void start() {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (m_enabled) {
      m_setup_start_time = std::chrono::steady_clock::now();
    }
#endif
  }

  /// Stops setup time measurement.
  void stop() {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (m_enabled) {
      m_setup_duration = std::chrono::steady_clock::now() - m_setup_start_time;
    }
#endif
  }

  /// Returns the setup duration in seconds.
  ///
  /// @return The setup duration in seconds.
  const std::chrono::duration<double>& duration() const {
    return m_setup_duration;
  }

 private:
  [[maybe_unused]]
  bool m_enabled = true;
  std::chrono::steady_clock::time_point m_setup_start_time;
  std::chrono::duration<double> m_setup_duration{0.0};
};

/// Records the number of tracer measurements (start/stop pairs) and the average
/// duration between each start and stop call.
class SolveTimeTracer {
 public:
  /// Constructs an enabled tracer.
  SolveTimeTracer() = default;

  /// Constructs a tracer.
  ///
  /// @param enabled Whether to measure time.
  explicit SolveTimeTracer(bool enabled) : m_enabled{enabled} {}

  /// Starts solve time measurement.
  void start() {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (m_enabled) {
      m_current_solve_start_time = std::chrono::steady_clock::now();
    }
#endif
  }

  /// Stops solve time measurement, increments the number of averages, and
  /// incorporates the latest measurement into the average.
  void stop() {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (m_enabled) {
      m_current_solve_duration =
          std::chrono::steady_clock::now() - m_current_solve_start_time;
      m_total_solve_duration += m_current_solve_duration;

      ++m_num_solves;
      m_average_solve_duration =
          (m_num_solves - 1.0) / m_num_solves * m_average_solve_duration +
          1.0 / m_num_solves * m_current_solve_duration;
    }
#endif
  }

  /// Returns the number of solves.
  ///
  /// @return The number of solves.
  int num_solves() const { return m_num_solves; }

  /// Returns the most recent solve duration in seconds.
  ///
  /// @return The most recent solve duration in seconds.
  const std::chrono::duration<double>& current_duration() const {
    return m_current_solve_duration;
  }

  /// Returns the average solve duration in seconds.
  ///
  /// @return The average solve duration in seconds.
  const std::chrono::duration<double>& average_duration() const {
    return m_average_solve_duration;
  }

  /// Returns the sum of all solve durations in seconds.
  ///
  /// @return The sum of all solve durations in seconds.
  const std::chrono::duration<double>& total_duration() const {
    return m_total_solve_duration;
  }

 private:
  [[maybe_unused]]
  bool m_enabled = true;
  std::chrono::steady_clock::time_point m_current_solve_start_time;
  std::chrono::duration<double> m_current_solve_duration{0.0};
  std::chrono::duration<double> m_total_solve_duration{0.0};

  int m_num_solves = 0;
  std::chrono::duration<double> m_average_solve_duration{0.0};
};

/// Time tracer concept.
template <typename T>
concept TimeTracer =
    std::same_as<T, SetupTimeTracer> || std::same_as<T, SolveTimeTracer>;

/// Starts a tracer in the constructor and stops it in the destructor.
///
/// @tparam Tracer Time tracer type.
template <TimeTracer Tracer>
class ScopedTimeTracer {
 public:
  /// Starts a tracer.
  ///
  /// @param tracer The tracer.
  explicit ScopedTimeTracer(Tracer& tracer) noexcept : m_tracer{&tracer} {
    m_tracer->start();
  }

  /// Stops a tracer.
  ~ScopedTimeTracer() {
    if (m_active) {
      m_tracer->stop();
    }
  }

  /// Move constructor.
  ///
  /// @param rhs The other ScopedTimeTracer.
  ScopedTimeTracer(ScopedTimeTracer&& rhs) noexcept
      : m_tracer{rhs.m_tracer}, m_active{rhs.m_active} {
    rhs.m_active = false;
  }

  ScopedTimeTracer(const ScopedTimeTracer&) = delete;
  ScopedTimeTracer& operator=(const ScopedTimeTracer&) = delete;

  /// Stops the tracer.
  ///
  /// If this is called, the destructor is a no-op.
  void stop() {
    if (m_active) {
      m_tracer->stop();
      m_active = false;
    }
  }

  /// Returns the tracer's most recent solve duration in seconds.
  ///
  /// @return The tracer's most recent solve duration in seconds.
  const std::chrono::duration<double>& current_duration() const {
    return m_tracer->current_duration();
  }

 private:
  Tracer* m_tracer;
  bool m_active = true;
};

/// Times a sequence of back-to-back phases.
///
/// Starting a phase stops the previous one, and the destructor stops the active
/// one, so every exit path closes the phase that was running.
///
/// @tparam Tracer Time tracer type.
template <TimeTracer Tracer>
class PhaseTimeTracer {
 public:
  /// Starts the first phase.
  ///
  /// @param tracer The first phase's tracer.
  explicit PhaseTimeTracer(Tracer& tracer) noexcept : m_tracer{&tracer} {
    m_tracer->start();
  }

  /// Stops the active phase.
  ~PhaseTimeTracer() { stop(); }

  PhaseTimeTracer(const PhaseTimeTracer&) = delete;
  PhaseTimeTracer& operator=(const PhaseTimeTracer&) = delete;

  /// Stops the active phase, if any, and starts the next one.
  ///
  /// @param tracer The next phase's tracer.
  void next(Tracer& tracer) {
    stop();
    m_tracer = &tracer;
    m_tracer->start();
  }

  /// Stops the active phase, if any.
  void stop() {
    if (m_tracer != nullptr) {
      m_tracer->stop();
      m_tracer = nullptr;
    }
  }

 private:
  Tracer* m_tracer;
};

/// Wraps a callable so each call is timed by the given tracer.
///
/// @tparam Tracer Time tracer type.
/// @tparam F Callable type.
/// @param tracer The tracer. It must outlive the returned callable.
/// @param f The callable. It must outlive the returned callable.
/// @return A callable that forwards its arguments to f while tracing.
template <TimeTracer Tracer, typename F>
auto traced(Tracer& tracer, const F& f) {
  return [&tracer, &f]<typename... Args>(Args&&... args) -> decltype(auto) {
    ScopedTimeTracer scoped_tracer{tracer};
    return f(std::forward<Args>(args)...);
  };
}

/// An entry in a time trace.
///
/// @tparam ID Time tracer ID enum type.
template <typename ID>
  requires std::is_enum_v<ID>
struct TimeTraceEntry {
  /// Time tracer ID.
  ID id;

  /// Name of measurement to show in diagnostics.
  std::string_view name;

  /// Nesting depth in the time trace. The root has depth 0.
  int depth;
};

namespace detail {

/// Returns true if the time trace has one root followed by its descendants and
/// lists each ID at most once, in increasing order.
///
/// @tparam ID Time tracer ID enum type.
/// @tparam N Number of time trace entries.
/// @param entries Time trace entries.
/// @return True if the time trace is valid.
template <typename ID, size_t N>
consteval bool is_valid_time_trace(
    const std::array<TimeTraceEntry<ID>, N>& entries) {
  if (N == 0 || entries[0].depth != 0) {
    return false;
  }

  for (size_t i = 1; i < N; ++i) {
    // Each entry after the root must have a larger ID than the previous entry
    if (std::to_underlying(entries[i].id) <=
        std::to_underlying(entries[i - 1].id)) {
      return false;
    }

    // Each entry after the root must be a descendant of the root
    if (entries[i].depth < 1) {
      return false;
    }

    // Each entry must be at most one level deeper than the previous entry, so
    // it's either the previous entry's child, its sibling, or a sibling of one
    // of its ancestors
    if (entries[i].depth > entries[i - 1].depth + 1) {
      return false;
    }
  }

  return true;
}

/// Called by TimeTracerIndex when a tracer ID isn't in the time trace.
///
/// This is never defined, so calling it at compile time is an error that names
/// the problem.
void tracer_id_is_not_in_time_trace();

}  // namespace detail

/// Index of a tracer in a time trace, found at compile time.
///
/// @tparam Entries Time trace entries.
template <const auto& Entries>
  requires(detail::is_valid_time_trace(Entries))
class TimeTracerIndex {
 public:
  /// Time tracer ID enum type.
  using ID = decltype(Entries[0].id);

  /// Finds the given tracer ID in the time trace.
  ///
  /// This is implicit so TimeTracerTree::operator[] can take a tracer ID. An ID
  /// that isn't in the time trace is a compile error.
  ///
  /// @param id Time tracer ID.
  // NOLINTNEXTLINE (google-explicit-constructor)
  consteval TimeTracerIndex(ID id) {
    for (size_t i = 0; i < Entries.size(); ++i) {
      if (Entries[i].id == id) {
        m_index = i;
        return;
      }
    }

    detail::tracer_id_is_not_in_time_trace();
  }

  /// Returns the tracer's index in the time trace.
  ///
  /// @return The tracer's index in the time trace.
  constexpr size_t value() const { return m_index; }

 private:
  size_t m_index = 0;
};

/// A set of tracers, one per entry in a time trace.
///
/// @tparam Tracer Time tracer type.
/// @tparam Entries Time trace entries. The first entry is the root, which
///     should time everything the other entries do.
template <TimeTracer Tracer, const auto& Entries>
  requires(detail::is_valid_time_trace(Entries))
class TimeTracerTree {
 public:
  /// Time tracer ID enum type.
  using ID = typename TimeTracerIndex<Entries>::ID;

  /// Constructs the tracers.
  ///
  /// @param enabled Whether the tracers measure time.
  explicit TimeTracerTree(bool enabled) { m_tracers.fill(Tracer{enabled}); }

  /// Returns the tracer with the given ID.
  ///
  /// @param index Time tracer ID, converted to its index at compile time. It
  ///     must be in the time trace.
  /// @return The tracer with the given ID.
  Tracer& operator[](TimeTracerIndex<Entries> index) {
    return m_tracers[index.value()];
  }

  /// Returns the root tracer.
  ///
  /// @return The root tracer.
  Tracer& root() { return m_tracers[0]; }

  /// Returns the time trace entries.
  ///
  /// @return The time trace entries.
  static constexpr std::span<const TimeTraceEntry<ID>, Entries.size()>
  entries() {
    return Entries;
  }

  /// Returns the tracers in time trace order.
  ///
  /// @return The tracers in time trace order.
  std::span<const Tracer, Entries.size()> tracers() const { return m_tracers; }

 private:
  std::array<Tracer, Entries.size()> m_tracers;
};

}  // namespace slp
