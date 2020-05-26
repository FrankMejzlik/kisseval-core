

#include <array>
#include <chrono>
#include <iostream>
#include <thread>

#include "common.h"
#include "utility.h"

#include "ImageRanker.h"

#include "Tester.hpp"

using namespace image_ranker;

int main()
{
  ImageRanker::Config cfg =
      ImageRanker::parse_data_config_file(ImageRanker::eMode::cFullAnalytical, DATA_INFO_FPTH, DATA_DIR);

  ImageRanker ranker(cfg);

  return 0;
}
