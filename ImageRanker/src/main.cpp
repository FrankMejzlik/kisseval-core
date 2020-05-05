

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

  // Statistics
#define TEST_get_search_sessions_rank_progress_chart_data 1
#define TEST_get_histogram_used_labels 0

#define TEST_submit_annotator_user_queries 0
#define TEST_submit_search_session 0
#define TEST_get_frame_detail_data 0

#define TEST_get_random_frame_sequence 0
#define TEST_get_autocomplete_results 0
#define TEST_get_loaded_imagesets_info 0
#define TEST_rank_frames 0
#define TEST_run_model_test 0
#define TEST_run_model_test_with_sim 0
#define TEST_run_model_vec_space 0
#define TEST_boolean_grid_test_threshold 0
#define TEST_run_model_test_Google 0
#define TEST_boolean_grid_test_threshold_Google 0
#define TEST_run_model_test_W2VV 0

#if TEST_get_search_sessions_rank_progress_chart_data
  {
    auto r11 = ranker.get_search_sessions_rank_progress_chart_data("NasNet2019", ""s);
    auto r1{ r11.aggregate_quantile_chart };
    std::cout << "xs: \t";
    for (auto&& val : r1.x)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;

    std::cout << "mins: \t";
    for (auto&& val : r1.y_min)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;

    std::cout << "Q1s: \t";
    for (auto&& val : r1.y_q1)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;

    std::cout << "Q2s: \t";
    for (auto&& val : r1.y_q2)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;

    std::cout << "Q3s: \t";
    for (auto&& val : r1.y_q3)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;

    std::cout << "maxes: \t";
    for (auto&& val : r1.y_max)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;
  }
#endif

#if TEST_get_histogram_used_labels
  {
    auto r1 = ranker.get_histogram_used_labels("NasNet2019", ""s, 100, true, 9);
    std::cout << "xs: \t";
    for (auto&& val : r1.x)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;

    std::cout << "fxs: \t";
    for (auto&& val : r1.fx)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    r1 = ranker.get_histogram_used_labels("NasNet2019", ""s, 100, false, 9);
    std::cout << "xs: \t";
    for (auto&& val : r1.x)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;

    std::cout << "fxs: \t";
    for (auto&& val : r1.fx)
    {
      std::cout << val << "\t";
    }
    std::cout << std::endl;
  }
#endif

  // TEST: `submit_annotator_user_queries`
#if TEST_submit_annotator_user_queries
  ranker.submit_annotator_user_queries(enum_label(eDataPackId::NASNET_2019).first, 9, true,
                                       {
                                           { "Shonicka1", "123&345&3232", "car,; '  '  \\ \\ cat, cow", 4321 },
                                           { "Shonicka2", "1213&3451&32321", "cars, cats, cows", 5321 },
                                       });
#endif

#if TEST_submit_search_session
  ranker.submit_search_session("NasNet2019",
                               "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_"
                               "op=mult;model_inner_op=max;transform=linear_01;sim_user=no_sim_user;",
                               10, true, 8888, eSearchSessionEndStatus::FOUND_TARGET, 12345, "jajsemsession1111",
                               std::vector<InteractiveSearchAction>{
                                   { 0, "add_from_autocomplete", "1234", "apple", 333, 4 },
                                   { 0, "delete_from_query", "1234", "apple", 88888, 10 },
                                   { 0, "add_from_detail", "234", "tuple", 12, 13 },
                               });
#endif

#if TEST_get_frame_detail_data
  /** Man on a bike in bright hall */
  auto res = ranker.get_frame_detail_data(
      1, "NasNet2019",
      "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_"
      "op=mult;model_inner_op=max;transform=linear_01;sim_user=no_sim_user;",
      true, false);

  for (auto&& p_kw : res.top_keywords)
  {
    std::cout << "ID = " << p_kw->ID << std::endl;
    std::cout << "\t word = " << p_kw->m_word << std::endl;
  }

  std::cout << "======================== " << std::endl;

  /** Man on a bike in bright hall */
  res = ranker.get_frame_detail_data(
      1, "NasNet2019",
      "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_"
      "op=mult;model_inner_op=max;transform=linear_01;sim_user=no_sim_user;",
      true, true);

  for (auto&& p_kw : res.top_keywords)
  {
    std::cout << "ID = " << p_kw->ID << std::endl;
    std::cout << "\t word = " << p_kw->m_word << std::endl;
  }

#endif

  // TEST: `get_random_frame_sequence`
#if TEST_get_random_frame_sequence
  {
    auto r1 = ranker.get_random_frame_sequence(enum_label(eImagesetId::V3C1_20K).first, 3);
  }
#endif

  // TEST: `get_autocomplete_results`
#if TEST_get_autocomplete_results
  {
    auto r1 =
        ranker.get_autocomplete_results("NasNet2019", "w", 20, true,
                                        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_"
                                        "outter_op=mult;model_inner_op=max;transform=linear_01;sim_user=no_sim_user;");

    r1 =
        ranker.get_autocomplete_results("NasNet2019", "we", 20, true,
                                        "model=mult-sum-max;model_operations=mult-max;model_ignore_treshold=0.00;model_"
                                        "outter_op=mult;model_inner_op=max;transform=linear_01;sim_user=no_sim_user;");

    if (r1.top_keywords.size() < 20)
    {
      throw std::runtime_error("failed");
    }
  }
#endif

  // TEST: `get_loaded_imagesets_info`
#if TEST_get_loaded_imagesets_info
  {
    auto r1 = ranker.get_loaded_imagesets_info();
    auto r2 = ranker.get_loaded_data_packs_info();
  }

#endif

  // TEST: `rank_frames`
#if TEST_rank_frames
  {
    std::vector<std::string> a = { "1&2", "3&4" };
    std::vector<std::string> b{ "1&2", "3&4" };

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
        { "&-274+" }, dp1["ID"].get<std::string>(),
        "model=mult-sum-max;transform=linear_01;model_operations=mult-sum;model_ignore_treshold=0.0", 1000, 1304);
  }
#endif

#if TEST_run_model_test
  {
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
  }

#endif

#if TEST_run_model_test_with_sim
  {
    std::cout << "===============================" << std::endl;
    std::cout << "MULT-SUM-MAX model: " << std::endl;
    std::cout << "\t Simulate single queries: " << std::endl;
    std::cout << "--------------" << std::endl;

    auto r1_opts(
        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_op=mult;model_inner_op=max;transform=linear_01;sim_user=user_model_x_to_p;sim_user_paremeter_p=6.0;sim_user_target=single_queries;sim_user_num_words_from=1;sim_user_num_words_to=8;"s);
    auto r1 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", r1_opts);
    auto r1_area = calc_chart_area(r1);
    std::cout << "\t "
              << "sim_user=user_model_x_to_p;sim_user_paremeter_p=6.0;sim_user_target=single_queries" << std::endl;
    std::cout << "\t\t area: " << r1_area << std::endl;

    // Assert results

    r1_opts =
        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_op=mult;model_inner_op=max;transform=linear_01;sim_user=user_model_x_to_p;sim_user_paremeter_p=6.0;sim_user_target=temp_queries;sim_user_num_words_from=1;sim_user_num_words_to=8;"s;
    r1 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", r1_opts);
    r1_area = calc_chart_area(r1);
    std::cout << "\t "
              << "sim_user=user_model_x_to_p;sim_user_paremeter_p=6.0;sim_user_target=temp_queries" << std::endl;
    std::cout << "\t\t area: " << r1_area << std::endl;

    // Assert results

    r1_opts =
        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_op=sum;model_inner_op=sum;transform=linear_01;sim_user=user_model_x_to_p;sim_user_paremeter_p=6.0;sim_user_target=alter_real_with_temporal;sim_user_num_words_from=1;sim_user_num_words_to=8;"s;
    r1 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", r1_opts);
    r1_area = calc_chart_area(r1);
    std::cout << "\t "
              << "sim_user=user_model_x_to_p;sim_user_paremeter_p=6.0;sim_user_target=alter_real_with_temporal"
              << std::endl;
    std::cout << "\t\t area: " << r1_area << std::endl;

    // Assert results
  }
#endif

#if TEST_run_model_vec_space

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

  constexpr size_t num_iters{ 100_z };
  constexpr float p_from{ 0.0005F };
  constexpr float p_to{ 0.0006F };

  constexpr float delta_it{ (p_to - p_from) / num_iters };

  float max_area{ 0.0F };
  float max_p_val{ std::numeric_limits<float>::quiet_NaN() };

  for (auto [param, i] = std::tuple{ p_from, 0_z }; param <= p_to; param += delta_it, ++i)
  {
    auto res =
        ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019",
                              "model=boolean;model_true_threshold=" + std::to_string(param) + ";transform=linear_01;");

    float area{ calc_chart_area(res) };

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

  constexpr size_t num_iters{ 100_z };
  constexpr float p_from{ 0.0001F };
  constexpr float p_to{ 0.001F };

  constexpr float delta_it{ (p_to - p_from) / num_iters };

  float max_area{ 0.0F };
  float max_p_val{ std::numeric_limits<float>::quiet_NaN() };

  for (auto [param, i] = std::tuple{ p_from, 0_z }; param <= p_to; param += delta_it, ++i)
  {
    auto res =
        ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "GoogleVisionAi_Sep2019",
                              "model=boolean;model_true_threshold=" + std::to_string(param) + ";transform=linear_01;");

    float area{ calc_chart_area(res) };

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

#if TEST_run_model_test_W2VV

  auto r1 = ranker.rank_frames(
      std::vector<std::string>{ "A man wearing a black t-shirt and pants riding a bike on a concrete surface." },
      "W2VV_BoW_Dec2019", "model=w2vv_bow_plain;", 1000, true, 498);

  auto r2 =
      ranker.rank_frames(std::vector<std::string>{ "An old black car riding in a desert with a few bushes nearby." },
                         "W2VV_BoW_Dec2019", "model=w2vv_bow_plain;", 1000, true, 16554);

  std::cout << "===============================" << std::endl;
  std::cout << "===============================" << std::endl;
  std::cout << "W2VV: " << std::endl;
  std::cout << "===============================" << std::endl;
  std::cout << "===============================" << std::endl;

  std::string m11_opts = "model=w2vv_bow_plain;"s;
  auto r11 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "W2VV_BoW_Dec2019", m11_opts, true, 100, true);
  auto r11_area = calc_chart_area(r11);
  std::cout << "========================================" << std::endl;
  std::cout << m11_opts << std::endl;
  std::cout << "\t" << r11_area << std::endl;

#endif

  auto succs = ranker.frame_successors("V3C1_20k", 999, 4_z);

  return 0;
}
