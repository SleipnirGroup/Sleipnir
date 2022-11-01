// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Indexer.h"

using namespace sleipnir::autodiff;

size_t Indexer::GetIndex() {
  return index++;
}
