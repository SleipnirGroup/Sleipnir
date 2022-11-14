// Copyright (c) Joshua Nichols and Tyler Veness

#include "Indexer.hpp"

using namespace sleipnir::autodiff;

size_t Indexer::GetIndex() {
  return index++;
}
