

#include <array>
#include <chrono>
#include <iostream>
#include <thread>

#define DATA_DIR "data/"
#define DATA_INFO_FPTH R"(c:\Users\devwe\data\data_info.json)"

#include "common.h"
#include "utility.h"

#include "ImageRanker.h"

using namespace image_ranker;

int main()
{
  auto json_data = parse_data_config_file(DATA_INFO_FPTH);

  auto is1 = json_data["imagesets"][0];
  auto dp1 = json_data["data_packs"]["VIRET_based"][0];
  auto dp2 = json_data["data_packs"]["VIRET_based"][1];

  // V3C1 20k subset
  std::vector<DatasetDataPackRef> datasets{
      {is1["ID"].get<std::string>(), is1["description"].get<std::string>(), is1["ID"].get<std::string>(),
       DATA_DIR + is1["frames_dir"].get<std::string>(), DATA_DIR + is1["ID_to_frame_fpth"].get<std::string>()}};

  std::vector<ViretDataPackRef> VIRET_data_packs{
      // NasNet
      {dp1["ID"].get<std::string>(), dp1["description"].get<std::string>(), dp1["model_options"].get<std::string>(),
       dp1["data"]["target_dataset"].get<std::string>(),

       dp1["vocabulary"]["ID"].get<std::string>(), dp1["vocabulary"]["description"].get<std::string>(),
       DATA_DIR + dp1["vocabulary"]["keyword_synsets_fpth"].get<std::string>(),

       DATA_DIR + dp1["data"]["presoftmax_scorings_fpth"].get<std::string>(),
       DATA_DIR + dp1["data"]["softmax_scorings_fpth"].get<std::string>(),
       DATA_DIR + dp1["data"]["deep_features_fpth"].get<std::string>()},
      // GoogLeNet
      {dp2["ID"].get<std::string>(), dp2["description"].get<std::string>(), dp2["model_options"].get<std::string>(),
       dp2["data"]["target_dataset"].get<std::string>(),

       dp2["vocabulary"]["ID"].get<std::string>(), dp2["vocabulary"]["description"].get<std::string>(),
       DATA_DIR + dp2["vocabulary"]["keyword_synsets_fpth"].get<std::string>(),

       DATA_DIR + dp2["data"]["presoftmax_scorings_fpth"].get<std::string>(),
       DATA_DIR + dp2["data"]["softmax_scorings_fpth"].get<std::string>(),
       DATA_DIR + dp2["data"]["deep_features_fpth"].get<std::string>()},
  };

    ImageRanker::Config cfg{ImageRanker::eMode::cFullAnalytical, datasets, VIRET_data_packs,
                            std::vector<GoogleDataPackRef>(), std::vector<BowDataPackRef>()};

  ImageRanker ranker(cfg);

#define TEST_submit_annotator_user_queries 0
#define TEST_get_random_frame_sequence 0
#define TEST_get_autocomplete_results 0
#define TEST_get_loaded_imagesets_info 0
#define TEST_rank_frames 0
#define TEST_run_model_test 1

  // TEST: `submit_annotator_user_queries`
#if TEST_submit_annotator_user_queries
  ranker.submit_annotator_user_queries(enum_label(eDataPackId::NASNET_2019).first, 9, true,
                                       {
                                           {"Shonicka1", "123&345&3232", "car,; '  '  \\ \\ cat, cow", 4321},
                                           {"Shonicka2", "1213&3451&32321", "cars, cats, cows", 5321},
                                       });
#endif

  // TEST: `get_random_frame_sequence`
#if TEST_get_random_frame_sequence
  auto r1 = ranker.get_random_frame_sequence(enum_label(eImagesetId::V3C1_20K).first, 3);
#endif

  // TEST: `get_autocomplete_results`
#if TEST_get_autocomplete_results
  auto r1 = ranker.get_autocomplete_results(enum_label(eDataPackId::NASNET_2019).first, "ca", 20, true);

  if (r1.top_keywords.size() < 20)
  {
    throw std::runtime_error("failed");
  }
#endif

  // TEST: `get_loaded_imagesets_info`
#if TEST_get_loaded_imagesets_info

  auto r1 = ranker.get_loaded_imagesets_info();
  auto r2 = ranker.get_loaded_data_packs_info();

#endif

  // TEST: `rank_frames`
#if TEST_rank_frames

  std::vector<std::string> a = {"1&2", "3&4"};
  std::vector<std::string> b{"1&2", "3&4"};

  /**
   * model_ID:
   *    "boolean"
   *    "vector_space"
   *    "mult-sum-max"
   *    "boolean_bucket"
   *
   * transform_ID:
   *    "no_transform"
   *    "linear_0-1"
   *    "softmax"
   *
   */

  auto r1 = ranker.rank_frames(
      {"&-274+"}, dp1["ID"].get<std::string>(),
      "model=mult-sum-max;transform=linear_01;model_operations=mult-sum;model_ignore_treshold=0.0", 1000, 1304);

#endif

#if TEST_run_model_test

  auto r1 = ranker.run_model_test(
      eUserQueryOrigin::SEMI_EXPERTS, dp1["ID"].get<std::string>(),
      "model=mult-sum-max;transform=linear_01;model_operations=mult-sum;model_ignore_treshold=0.0");

#endif

  return 0;
}
