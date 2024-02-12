// Copyright (c) Sleipnir contributors

#include <catch2/catch_session.hpp>

int main(int argc, char* argv[]) {
  Catch::Session session;
  session.configData().allowZeroTests = true;

  return session.run(argc, argv);
}
