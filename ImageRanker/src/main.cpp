

#include <array>
#include <chrono>
#include <iostream>
#include <thread>

#define DATA_DIR "data/"
#define DATA_INFO_FPTH R"(c:\Users\devwe\source\repos\ImageRanker\data_info.json)"

#include "common.h"
#include "utility.h"

#include "ImageRanker.h"

using namespace image_ranker;

int main()
{
  ImageRanker::Config cfg =
      ImageRanker::parse_data_config_file(ImageRanker::eMode::cFullAnalytical, DATA_INFO_FPTH, DATA_DIR);

  ImageRanker ranker(cfg);

#define TEST_submit_annotator_user_queries 0
#define TEST_get_random_frame_sequence 0
#define TEST_get_autocomplete_results 0
#define TEST_get_loaded_imagesets_info 0
#define TEST_rank_frames 0
#define TEST_run_model_test 0
#define TEST_run_model_vec_space 0
#define TEST_boolean_grid_test_threshold 0
#define TEST_run_model_test_Google 0
#define TEST_boolean_grid_test_threshold_Google 1

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

  std::cout << "===============================" << std::endl;
  std::cout << "MULT-SUM-MAX model: " << std::endl;
  std::cout << "--------------" << std::endl;
  auto r1 = ranker.run_model_test(
      eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019",
      "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_op=sum;model_inner_op="
      "sum;transform=linear_01;sim_user=no_sim_user;");
  auto r1_area = calc_chart_area(r1);
  std::cout << "transform=linear_01, model_operations=mult-sum: " << r1_area << std::endl;

  std::cout << "===============================" << std::endl;
  std::cout << "Boolean model: " << std::endl;
  std::cout << "--------------" << std::endl;

  auto r2 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019",
                                  "model=boolean;model_true_threshold=0.000598001;transform=linear_01;");
  auto r2_area = calc_chart_area(r2);
  std::cout << "t = 0.000598001: " << r2_area << std::endl;

  std::cout << "===============================" << std::endl;
  std::cout << "Vector space model: " << std::endl;
  std::cout << "--------------" << std::endl;

#endif

#if TEST_run_model_vec_space
  /*std::string m6_opts =
      "model=vector_space;transform=linear_01;model_dist_fn=cosine;model_term_tf=natural;model_term_idf=idf;model_query_tf=augmented;model_query_idf=none;model_true_threshold=0.000698001;"s;*/

  /*std::string m4_opts =
      "model=vector_space;transform=linear_01;model_dist_fn=cosine;model_term_tf=natural;model_term_idf=idf;model_query_tf=augmented;model_query_idf=none;model_true_threshold=0.002;"s;
  auto r4 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", m4_opts);
  auto r4_area = calc_chart_area(r4);
  std::cout << m4_opts << std::endl;
  std::cout << "\t" << r4_area << std::endl;*/

  std::string m5_opts =
      "model=vector_space;transform=linear_01;model_dist_fn=cosine;model_term_tf=natural;model_term_idf=idf;model_query_tf=log;model_query_idf=none;model_idf_coef=6.0f"s;
  auto r5 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", m5_opts);
  auto r5_area = calc_chart_area(r5);
  std::cout << m5_opts << std::endl;
  std::cout << "\t" << r5_area << std::endl;

  std::string m6_opts =
      "model=vector_space;transform=linear_01;model_dist_fn=cosine;model_term_tf=natural;model_term_idf=idf;model_query_tf=log;model_query_idf=idf;model_idf_coef=6.0f"s;
  auto r6 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", m6_opts);
  auto r6_area = calc_chart_area(r6);
  std::cout << m6_opts << std::endl;
  std::cout << "\t" << r6_area << std::endl;

  std::string m7_opts =
      "model=vector_space;transform=linear_01;model_dist_fn=cosine;model_term_tf=natural;model_term_idf=idf;model_query_tf=augmented;model_query_idf=idf;model_idf_coef=6.0f"s;
  auto r7 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", m7_opts);
  auto r7_area = calc_chart_area(r7);
  std::cout << m7_opts << std::endl;
  std::cout << "\t" << r7_area << std::endl;

#endif

#if TEST_boolean_grid_test_threshold

  constexpr size_t num_iters{100_z};
  constexpr float p_from{0.0005F};
  constexpr float p_to{0.0006F};

  constexpr float delta_it{(p_to - p_from) / num_iters};

  float max_area{0.0F};
  float max_p_val{std::numeric_limits<float>::quiet_NaN()};

  for (auto [param, i] = std::tuple{p_from, 0_z}; param <= p_to; param += delta_it, ++i)
  {
    auto res =
        ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019",
                              "model=boolean;model_true_threshold=" + std::to_string(param) + ";transform=linear_01;");

    float area{calc_chart_area(res)};

    if (area > max_area)
    {
      std::cout << "New max found... p = " << param << ", i = " << i << std::endl;
      std::cout << "\t area = " << area << std::endl;
      max_area = area;
      max_p_val = param;
    }
    std::cout << "i = " << i << std::endl;
    std::cout << "\t area = " << area << std::endl;
  }

#endif

#if TEST_run_model_test_Google

  std::cout << "===============================" << std::endl;
  std::cout << "===============================" << std::endl;
  std::cout << "Google: " << std::endl;
  std::cout << "===============================" << std::endl;
  std::cout << "===============================" << std::endl;

  std::string m1_opts =
      "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_op=sum;model_inner_op=sum;transform=linear_01;sim_user=no_sim_user;"s;
  auto r1 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "GoogleVisionAi_Sep2019", m1_opts);
  auto r1_area = calc_chart_area(r1);
  std::cout << "========================================" << std::endl;
  std::cout << m1_opts << std::endl;
  std::cout << "\t" << r1_area << std::endl;

  std::string m2_opts = "sim_user=no_sim_user;model=boolean;model_true_threshold=0.000598001;transform=linear_01;"s;
  auto r2 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "GoogleVisionAi_Sep2019", m2_opts);
  auto r2_area = calc_chart_area(r2);
  std::cout << "========================================" << std::endl;
  std::cout << m2_opts << std::endl;
  std::cout << "\t" << r2_area << std::endl;

  std::string m5_opts =
      "sim_user=no_sim_user;model=vector_space;transform=linear_01;model_dist_fn=cosine;model_term_tf=natural;model_term_idf=idf;model_query_tf=log;model_query_idf=none;model_IDF_method_idf_coef=6.0f"s;
  auto r5 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "GoogleVisionAi_Sep2019", m5_opts);
  auto r5_area = calc_chart_area(r5);
  std::cout << "========================================" << std::endl;
  std::cout << m5_opts << std::endl;
  std::cout << "\t" << r5_area << std::endl;

#endif

#if TEST_boolean_grid_test_threshold_Google

  constexpr size_t num_iters{100_z};
  constexpr float p_from{0.0001F};
  constexpr float p_to{0.001F};

  constexpr float delta_it{(p_to - p_from) / num_iters};

  float max_area{0.0F};
  float max_p_val{std::numeric_limits<float>::quiet_NaN()};

  for (auto [param, i] = std::tuple{p_from, 0_z}; param <= p_to; param += delta_it, ++i)
  {
    auto res =
        ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "GoogleVisionAi_Sep2019",
                              "model=boolean;model_true_threshold=" + std::to_string(param) + ";transform=linear_01;");

    float area{calc_chart_area(res)};

    if (area > max_area)
    {
      std::cout << "New max found... p = " << param << ", i = " << i << std::endl;
      std::cout << "\t area = " << area << std::endl;
      max_area = area;
      max_p_val = param;
    }
    std::cout << "i = " << i << std::endl;
    std::cout << "\t area = " << area << std::endl;
  }

#endif

  return 0;
}
