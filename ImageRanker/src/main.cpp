/**
 * Entry point of this evaluation library if used as standalone project.
 *
 * This is where you can test it's public API or use it to
 * handle some one-time jobs for you.
 */

#include <iostream>

#include "common.h"
#include "utility.h"

#include "ImageRanker.h"
#include "Tester.hpp"

using namespace image_ranker;

int main()
{
  // Parse config JSON file
  ImageRanker::Config cfg =
      ImageRanker::parse_data_config_file(ImageRanker::eMode::cFullAnalytical, DATA_INFO_FPTH, DATA_DIR);

  // Instantiate ranker with parsed config
  ImageRanker ranker(cfg);

#if RUN_BASIC_TESTS
  Tester::test_public_methods(ranker, cfg);
#endif

  return 0;
}
