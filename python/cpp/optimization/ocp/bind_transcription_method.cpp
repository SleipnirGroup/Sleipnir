// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/optimization/ocp/transcription_method.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_transcription_method(nb::enum_<TranscriptionMethod>& e) {
  e.value("DIRECT_TRANSCRIPTION", TranscriptionMethod::DIRECT_TRANSCRIPTION,
          DOC(slp, TranscriptionMethod, DIRECT_TRANSCRIPTION));
  e.value("DIRECT_COLLOCATION", TranscriptionMethod::DIRECT_COLLOCATION,
          DOC(slp, TranscriptionMethod, DIRECT_COLLOCATION));
  e.value("SINGLE_SHOOTING", TranscriptionMethod::SINGLE_SHOOTING,
          DOC(slp, TranscriptionMethod, SINGLE_SHOOTING));
}

}  // namespace slp
