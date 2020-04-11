

#include <array>
#include <chrono>
#include <iostream>
#include <thread>
#include "ImageRanker.h"

#define BASE_DIR "./data/"

using namespace image_ranker;

int main()
{
  // V3C1 20k subset
  std::vector<DatasetDataPackRef> datasets{
      {enum_label(eImagesetId::V3C1_20K).first, enum_label(eImagesetId::V3C1_20K).second,
       enum_label(eImagesetId::V3C1_20K).first, "./data/thumbs", "./data/files.txt"}};

  std::vector<ViretDataPackRef> VIRET_data_packs{
      // NasNet
      {enum_label(eDataPackId::NASNET_2019).first, enum_label(eDataPackId::NASNET_2019).second,
       enum_label(eImagesetId::V3C1_20K).first,

       enum_label(eVocabularyId::VIRET_WORDNET_2019).first, enum_label(eVocabularyId::VIRET_WORDNET_2019).second,
       "./data/VIRET/keyword_classes.txt",

       "./data/VIRET/NasNet2019/Studenti_NasNetLarge.pre-softmax",
       "./data/VIRET/NasNet2019/Studenti_NasNetLarge.softmax",
       "./data/VIRET/NasNet2019/Studenti_NasNetLarge.deep-features"},
      // GoogLeNet
      {enum_label(eDataPackId::GOOGLENET_2019).first, enum_label(eDataPackId::GOOGLENET_2019).second,
       enum_label(eImagesetId::V3C1_20K).first,

       enum_label(eVocabularyId::VIRET_WORDNET_2019).first, enum_label(eVocabularyId::VIRET_WORDNET_2019).second,
       "./data/VIRET/keyword_classes.txt",

       "./data/VIRET/GoogLeNet2019/05_googlenet.pre-softmax", "./data/VIRET/GoogLeNet2019/05_googlenet.softmax",
       "./data/VIRET/GoogLeNet2019/05_googlenet.deep-features"}};

  ImageRanker::Config cfg{ImageRanker::eMode::cFullAnalytical, datasets, VIRET_data_packs,
                          std::vector<GoogleDataPackRef>(), std::vector<BowDataPackRef>()};

  ImageRanker ranker(cfg);

#define TEST_submit_annotator_user_queries 0
#define TEST_get_random_frame_sequence 0
#define TEST_get_autocomplete_results 0
#define TEST_get_loaded_imagesets_info 0
#define TEST_rank_frames 0

  

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

  // "model_ID=mult-sum-max;transform_ID=linear_0-1;model_outter=mult;model_inner=sum;model_ignore_treshold=0.0",
  auto r1 = ranker.rank_frames(
      {"1&2", "3&4"}, enum_label(eDataPackId::NASNET_2019).first,
      "model_ID=mult-sum-max;transform_ID=linear_0-1;model_outter=mult;model_inner=sum;model_ignore_treshold=0.0",
      1000);

#endif

  return 0;
}
