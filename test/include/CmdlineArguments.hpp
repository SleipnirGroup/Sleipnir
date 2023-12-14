// Copyright (c) Sleipnir contributors

#pragma once

#include <span>
#include <string_view>

/// Test commandline argument that enables solver diagnostics
inline constexpr std::string_view kEnableDiagnostics = "--enable-diagnostics";

/**
 * Initializes the commandline argument list for tests to retreive.
 */
void SetCmdlineArgs(char* argv[], int argc);

/**
 * Returns the test executable's commandline arguments.
 */
std::span<char*> GetCmdlineArgs();

/**
 * Returns true if the given argument is present in the test executable's
 * commandline arguments.
 */
bool CmdlineArgPresent(std::string_view arg);
