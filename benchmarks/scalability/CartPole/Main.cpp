// Copyright (c) Joshua Nichols and Tyler Veness

#include "CasADi.hpp"
#include "Sleipnir.hpp"
#include "Util.hpp"

int main() {
  constexpr bool diagnostics = false;

  constexpr auto T = 5_s;
  constexpr int kMinPower = 2;
  constexpr int kMaxPower = 3;

  return RunBenchmarksAndLog(diagnostics, T, kMinPower, kMaxPower,
                             &CartPoleCasADi, &CartPoleSleipnir);
}
