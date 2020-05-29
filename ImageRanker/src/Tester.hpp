#pragma once

#include "common.h"
#include "utility.h"

#include "DataManager.h"
#include "Database.h"
#include "ImageRanker.h"

using namespace image_ranker;

namespace image_ranker
{
class [[nodiscard]]
Tester{ /*
         * Methods
         */
        public : static void test_public_methods(ImageRanker & ranker, const ImageRanker::Config& cfg){
            /*
             * Data pack specific methods
             */
            for (auto&& pack
                 : cfg.VIRET_packs){ if (pack.ID == "NasNet2019"){ test_data_pack__NasNet2019(ranker);
}
else if (pack.ID == "GoogLeNet2019") { test_data_pack__GoogLeNet2019(ranker); }
else if (pack.ID == "ITECTiny_NasNet2019") { test_data_pack__ITECTiny_NasNet2019(ranker); }
}

for (auto&& pack : cfg.Google_packs)
{
  if (pack.ID == "GoogleVisionAi_Sep2019")
  {
    test_data_pack__GoogleVisionAi_Sep2019(ranker);
  }
}

for (auto&& pack : cfg.W2VV_packs)
{
  if (pack.ID == "W2VV_BoW_Dec2019")
  {
    test_data_pack__W2VV_BoW_Dec2019(ranker);
  }
  else if (pack.ID == "ITECTiny_W2VV_BoW_Dec2019")
  {
    test_data_pack__W2VV_BoW_Dec2019(ranker);
  }
}

/*
 * Other methods
 */
test_general_methods(ranker);
}

/**
 * Runs test that try different true thresholds to find the approximation ot the best one.
 *
 * \param Reference to ranker instance.
 */
static void grid_test_find_true_thresholds(ImageRanker& ranker)
{
  std::cout << "===========================================================" << std::endl;
  std::cout << "       Running Bool threshold grid search " << std::endl;
  std::cout << "===========================================================" << std::endl;
  {
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "\t NASNet, Linear, Boolean" << std::endl;
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

    constexpr size_t num_iters{ 50_z };
    constexpr float p_from{ 0.0005F };
    constexpr float p_to{ 0.00065F };

    constexpr float delta_it{ (p_to - p_from) / num_iters };

    float max_area{ 0.0F };
    float max_p_val{ std::numeric_limits<float>::quiet_NaN() };

    for (auto [param, i] = std::tuple{ p_from, 0_z }; param <= p_to; param += delta_it, ++i)
    {
      auto res = ranker.run_model_test(
          eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019",
          "model=boolean;model_true_threshold=" + std::to_string(param) + ";transform=linear_01;");

      float area{ calc_chart_area(res) };

      if (area > max_area)
      {
        std::cout << "New max found... p = " << param << ", i = " << i << std::endl;
        std::cout << "\t area = " << area << std::endl;
        max_area = area;
        max_p_val = param;
      }
      std::cout << "i = " << i << "\t area = " << area << std::endl;

      std::cout << "--- " << std::endl;
      std::cout << "MAX FOUND: " << std::endl;
      std::cout << "\t area = " << max_area << std::endl;
      std::cout << "\t t = " << max_p_val << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    }
  }

  {
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "\t NASNet, softmax, Boolean" << std::endl;
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

    constexpr size_t num_iters{ 50_z };
    constexpr float p_from{ 0.0005F };
    constexpr float p_to{ 0.00065F };

    constexpr float delta_it{ (p_to - p_from) / num_iters };

    float max_area{ 0.0F };
    float max_p_val{ std::numeric_limits<float>::quiet_NaN() };

    for (auto [param, i] = std::tuple{ p_from, 0_z }; param <= p_to; param += delta_it, ++i)
    {
      auto res =
          ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019",
                                "model=boolean;model_true_threshold=" + std::to_string(param) + ";transform=softmax;");

      float area{ calc_chart_area(res) };

      if (area > max_area)
      {
        std::cout << "New max found... p = " << param << ", i = " << i << std::endl;
        std::cout << "\t area = " << area << std::endl;
        max_area = area;
        max_p_val = param;
      }
      std::cout << "i = " << i << "\t area = " << area << std::endl;

      std::cout << "--- " << std::endl;
      std::cout << "MAX FOUND: " << std::endl;
      std::cout << "\t area = " << max_area << std::endl;
      std::cout << "\t t = " << max_p_val << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    }
  }

  {
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "\t GoogLeNet, Linear, Bool" << std::endl;
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

    constexpr size_t num_iters{ 50_z };
    constexpr float p_from{ 0.0005F };
    constexpr float p_to{ 0.0006F };

    constexpr float delta_it{ (p_to - p_from) / num_iters };

    float max_area{ 0.0F };
    float max_p_val{ std::numeric_limits<float>::quiet_NaN() };

    for (auto [param, i] = std::tuple{ p_from, 0_z }; param <= p_to; param += delta_it, ++i)
    {
      auto res = ranker.run_model_test(
          eUserQueryOrigin::SEMI_EXPERTS, "GoogLeNet2019",
          "model=vector_space;model_term_tf=natural;model_term_idf=idf;model_query_tf=augmented;model_query_idf=idf;"
          "model_dist_fn=cosine;model_IDF_method=static_threshold;model_IDF_method_true_threshold=" +
              std::to_string(param) + ";transform=linear_01;sim_user=no_sim_user;");

      float area{ calc_chart_area(res) };

      if (area > max_area)
      {
        std::cout << "New max found... p = " << param << ", i = " << i << std::endl;
        std::cout << "\t area = " << area << std::endl;
        max_area = area;
        max_p_val = param;
      }
      std::cout << "i = " << i << "\t area = " << area << std::endl;

      std::cout << "--- " << std::endl;
      std::cout << "MAX FOUND: " << std::endl;
      std::cout << "\t area = " << max_area << std::endl;
      std::cout << "\t t = " << max_p_val << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    }
  }

  {
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "\t GoogLeNet, softmax, Bool" << std::endl;
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

    constexpr size_t num_iters{ 100_z };
    constexpr float p_from{ 0.0000001F };
    constexpr float p_to{ 0.0001F };

    constexpr float delta_it{ (p_to - p_from) / num_iters };

    float max_area{ 0.0F };
    float max_p_val{ std::numeric_limits<float>::quiet_NaN() };

    for (auto [param, i] = std::tuple{ p_from, 0_z }; param <= p_to; param += delta_it, ++i)
    {
      auto res = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "GoogLeNet2019",
                                       "model=boolean;model_true_threshold=" + std::to_string(param) +
                                           ";model_IDF_method_idf_coef=6;transform=softmax;sim_user=no_sim_user;");

      float area{ calc_chart_area(res) };

      if (area > max_area)
      {
        std::cout << "New max found... p = " << param << ", i = " << i << std::endl;
        std::cout << "\t area = " << area << std::endl;
        max_area = area;
        max_p_val = param;
      }
      std::cout << "i = " << i << "\t area = " << area << std::endl;

      std::cout << "--- " << std::endl;
      std::cout << "MAX FOUND: " << std::endl;
      std::cout << "\t area = " << max_area << std::endl;
      std::cout << "\t t = " << max_p_val << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    }
  }
}

/**
 * Tests basic functionality of `NasNet2019` data pack methods.
 *
 * \param Reference to ranker instance.
 */
static void test_data_pack__NasNet2019(ImageRanker& ranker)
{
  LOGP("===========================================================");
  LOGP("    Running `NasNet2019` basic tests ");
  LOGP("===========================================================");

  {
    /** Man on a bike in bright hall */
    auto res = ranker.get_frame_detail_data(
        1, "NasNet2019",
        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_"
        "op=mult;model_inner_op=max;transform=linear_01;sim_user=no_sim_user;",
        true, true);

    /*for (auto&& p_kw : res.top_keywords)
    {
      std::cout << "ID = " << p_kw->ID << std::endl;
      std::cout << "\t word = " << p_kw->m_word << std::endl;
    }*/
  }

  /*
   * Get random sequence
   */
  {
    auto succs = ranker.frame_successors("V3C1_20k", 999, 4_z);
  }
  {
    auto r1 = ranker.get_random_frame_sequence("V3C1_20k", 3);
  }

  /*
   * Annotator queries
   */

  /*
   * Run model tests
   */
  {
    std::string m6_opts =
        "model=vector_space;transform=linear_01;model_dist_fn=cosine;model_term_tf=natural;model_term_idf=idf;model_query_tf=log;model_query_idf=idf;model_IDF_method_idf_coef=6.0f"s;
    auto r6 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", m6_opts);

    std::string m7_opts =
        "model=vector_space;transform=linear_01;model_dist_fn=cosine;model_term_tf=natural;model_term_idf=idf;model_query_tf=augmented;model_query_idf=idf;model_IDF_method_idf_coef=6.0f"s;
    auto r7 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", m7_opts);
  }
  {
    auto r1 = ranker.run_model_test(
        eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019",
        "model=mult-sum-max;transform=softmax;model_operations=mult-max;model_ignore_treshold=0.0");

    r1 = ranker.run_model_test(
        eUserQueryOrigin::SEMI_EXPERTS, "GoogleVisionAi_Sep2019",
        "model=mult-sum-max;transform=softmax;model_operations=mult-max;model_ignore_treshold=0.0");

    auto r2 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019",
                                    "model=boolean;model_true_threshold=0.000598001;transform=linear_01;");
  }

  /*
   * Test user simulations
   */
  {
    auto r1_opts(
        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_op=mult;model_inner_op=max;transform=linear_01;sim_user=user_model_x_to_p;sim_user_paremeter_p=6.0;sim_user_target=single_queries;sim_user_num_words_from=1;sim_user_num_words_to=8;"s);
    auto r1 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", r1_opts);

    // Assert results

    r1_opts =
        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_op=mult;model_inner_op=max;transform=linear_01;sim_user=user_model_x_to_p;sim_user_paremeter_p=6.0;sim_user_target=temp_queries;sim_user_num_words_from=1;sim_user_num_words_to=8;"s;
    r1 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", r1_opts);

    // Assert results

    r1_opts =
        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_op=sum;model_inner_op=sum;transform=linear_01;sim_user=user_model_x_to_p;sim_user_paremeter_p=6.0;sim_user_target=alter_real_with_temporal;sim_user_num_words_from=1;sim_user_num_words_to=8;"s;
    r1 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019", r1_opts);

    // Assert results
  }
  /*
   * Search session median chart data
   */
  {
    auto r11 = ranker.get_search_sessions_rank_progress_chart_data("NasNet2019", ""s);
    auto r1{ r11.aggregate_quantile_chart };

    /*std::cout << "xs: \t";
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
    std::cout << std::endl;*/
  }

  /*
   * Histogram data
   */
  {
    auto r1 = ranker.get_histogram_used_labels("NasNet2019", ""s, 100, true, 9);
    /*std::cout << "xs: \t";
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
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;*/

    r1 = ranker.get_histogram_used_labels("NasNet2019", ""s, 100, false, 9);
    /*std::cout << "xs: \t";
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
    std::cout << std::endl;*/
  }
}

/**
 * Tests basic functionality of `GoogLeNet2019` data pack methods.
 *
 * \param Reference to ranker instance.
 */
static void test_data_pack__GoogLeNet2019([[maybe_unused]] ImageRanker& ranker)
{
  LOGP("===========================================================");
  LOGP("    Running `GoogLeNet2019` basic tests ");
  LOGP("===========================================================");
}

/**
 * Tests basic functionality of `ITECTiny_NasNet2019` data pack methods.
 *
 * \param Reference to ranker instance.
 */
static void test_data_pack__ITECTiny_NasNet2019(ImageRanker& ranker)
{
  LOGP("===========================================================");
  LOGP("    Running `ITECTiny_NasNet2019` basic tests ");
  LOGP("===========================================================");

  const auto data_pack_ID{ "ITECTiny_NasNet2019"s };
  const auto vocabulary_ID{ "VIRET_WordNet2019"s };
  const size_t user_level{ 10 };
  const DataManager& dm{ ranker._data_manager };

  /*
   * Submission of user queries
   */
  {
    const std::string tname{ data_pack_ID + ": User query submit" };
    LOGP("-------------------------------");
    LOGP("Running test: " + tname);

    auto qs{ dm.fetch_user_test_queries(eUserQueryOrigin::SEMI_EXPERTS, vocabulary_ID, data_pack_ID, ""s) };
    size_t num_queries_pre{ qs.size() };

    const size_t num_submits{ rand(1_z, 5_z) };

    // Do the inserting
    for (size_t i{ 0 }; i < num_submits; ++i)
    {
      ranker.submit_annotator_user_queries(
          data_pack_ID, "some_options", user_level, true,
          std::vector<AnnotatorUserQuery>{
              AnnotatorUserQuery{ "test_submit_" + std::to_string(i), std::vector<std::string>{ "123&345&3232" },
                                  std::vector<std::string>{ "car, cat, cow" }, std::vector<FrameId>{ 123 } } });
    }

    auto qs_post{ dm.fetch_user_test_queries(eUserQueryOrigin::SEMI_EXPERTS, vocabulary_ID, data_pack_ID, ""s) };
    size_t num_queries_post{ qs_post.size() };

    // Check post-condition
    if (num_queries_post != num_queries_pre + num_submits)
    {
      LOGP("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      LOGP("!!!!!!! Test failed !!!!!!!!!!!");
      LOGP("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }
    else
    {
      LOGP("+++++++++++++++++++++++++++++++");
      LOGP("TEST OK");
      LOGP("+++++++++++++++++++++++++++++++");
    }
  }
  /*
   * Search sessions
   */
  {
    const std::string tname{ "TEST '" + data_pack_ID + "': Search session submit" };
    LOGP("-------------------------------");
    LOGP("Running test: " + tname);

    // Get search sessions' data
    auto [search_sessions_pre, actions_pre]{ dm.fetch_search_sessions(data_pack_ID, user_level) };
    size_t num_ss_pre{ search_sessions_pre.size() };
    size_t num_actions_pre{ actions_pre.size() };

    const size_t num_submits{ rand(1_z, 5_z) };
    size_t total_num_actions{ 0 };

    // Do the inserting
    for (size_t i{ 0 }; i < num_submits; ++i)
    {
      const size_t num_actions{ rand(1_z, 10_z) };
      total_num_actions += num_actions;

      // Generate actions
      std::vector<InteractiveSearchAction> actions;
      for (size_t j{ 0 }; j < num_actions; ++j)
      {
        actions.emplace_back(InteractiveSearchAction{ 0, "add_from_autocomplete", "1234", "apple", j, j * 100, true });
      }

      ranker.submit_search_session(
          data_pack_ID,
          "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.0;model_outter_"
          "op=mult;model_inner_op=max;transform=linear_01;sim_user=no_sim_user;",
          user_level, true, 8888, eSearchSessionEndStatus::FOUND_TARGET, 12345, "test_submit_" + std::to_string(i),
          actions);
    }

    auto [search_sessions_post, actions_post]{ dm.fetch_search_sessions(data_pack_ID, user_level) };
    size_t num_ss_post{ search_sessions_post.size() };
    size_t num_actions_post{ actions_post.size() };

    // Check post-condition
    if (num_ss_post != num_ss_pre + num_submits && num_actions_post != num_actions_pre + total_num_actions)
    {
      LOGP("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      LOGP("!!!!!!! Test failed !!!!!!!!!!!");
      LOGP("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }
    else
    {
      LOGP("+++++++++++++++++++++++++++++++");
      LOGP("TEST OK");
      LOGP("+++++++++++++++++++++++++++++++");
    }
  }

  /*
   * Get autocomplete.
   */
  {
    const std::string tname{ "TEST '" + data_pack_ID + "': get_autocomplete_results" };
    LOGP("-------------------------------");
    LOGP("Running test: " + tname);

    auto r1{ ranker.get_autocomplete_results(
        data_pack_ID, "cat furniture", 20, true,
        "model=mult-sum-max;transform=softmax;model_operations=mult-sum;model_ignore_treshold=0.0") };

    auto r2{ ranker.get_autocomplete_results(
        data_pack_ID, "we", 20, true,
        "model=mult-sum-max;transform=linear_01;model_operations=mult-sum;model_ignore_treshold=0.0") };

    // Check post-condition
    if (true)
    {
      LOGP("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      LOGP("!!!!!!! Test failed !!!!!!!!!!!");
      LOGP("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }
    else
    {
      LOGP("+++++++++++++++++++++++++++++++");
      LOGP("TEST OK");
      LOGP("+++++++++++++++++++++++++++++++");
    }
  }

  /*
   * Run model tests
   */
  {
    auto rr{ ranker.rank_frames(std::vector<std::string>{ "Duck." }, "ITECTiny_W2VV_BoW_Dec2019",
                                "model=w2vv_bow_plain;", 10, true, 504) };

    auto re{ ranker.rank_frames(std::vector<std::string>{ "Duck." }, "W2VV_BoW_Dec2019", "model=w2vv_bow_plain;", 10,
                                true, 504) };

    // std::string m12_opts =
    //    "model=w2vv_bow_plain;";
    // auto r21 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, data_pack_ID, m12_opts, true);

    std::string m1_opts =
        "model=boolean;model_true_threshold=0.000598;model_IDF_method_idf_coef=6;transform=linear_01;sim_user=no_sim_"
        "user;";
    auto r1 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, data_pack_ID, m1_opts);

    std::string m2_opts =
        "model=vector_space;model_term_tf=natural;model_term_idf=idf;model_query_tf=augmented;model_query_idf=idf;"
        "model_dist_fn=cosine;model_IDF_method=score_adding;model_IDF_method_idf_coef=6;transform=linear_01;sim_user="
        "no_sim_user;";
    auto r2 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, data_pack_ID, m2_opts);

    std::string m3_opts =
        "model=mult-sum-max;model_ignore_treshold=0;model_operations=mult-sum;model_outter_op=mult;model_inner_op="
        "max;transform=linear_01;sim_user=no_sim_user;";
    auto r3 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, data_pack_ID, m3_opts);
  }
}

/**
 * Tests basic functionality of `GoogleVisionAi_Sep2019` data pack methods.
 *
 * \param Reference to ranker instance.
 */
static void test_data_pack__GoogleVisionAi_Sep2019(ImageRanker& ranker)
{
  LOGP("===========================================================");
  LOGP("    Running `GoogleVisionAi_Sep2019` basic tests ");
  LOGP("===========================================================");

  {
    /** Man on a bike in bright hall */
    auto res = ranker.get_frame_detail_data(
        1, "GoogleVisionAi_Sep2019",
        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_"
        "op=mult;model_inner_op=max;transform=linear_01;sim_user=no_sim_user;",
        true, false);

    /*for (auto&& p_kw : res.top_keywords)
    {
      std::cout << "ID = " << p_kw->ID << std::endl;
      std::cout << "\t word = " << p_kw->m_word << std::endl;
    }*/
  }

  /*
   * Autocomplete
   */
  {
    auto r1 = ranker.get_autocomplete_results(
        "GoogleVisionAi_Sep2019", "cat furniture", 20, true,
        "model=mult-sum-max;transform=softmax;model_operations=mult-sum;model_ignore_treshold=0.0");

    r1 = ranker.get_autocomplete_results(
        "GoogleVisionAi_Sep2019", "we", 20, true,
        "model=mult-sum-max;transform=linear_01;model_operations=mult-sum;model_ignore_treshold=0.0");

    if (r1.top_keywords.size() < 20)
    {
      throw std::runtime_error("failed");
    }
  }
  {
    std::string m1_opts =
        "model=mult-sum-max;model_operations=mult-sum;model_ignore_treshold=0.00;model_outter_op=sum;model_inner_op=sum;transform=linear_01;sim_user=no_sim_user;"s;
    auto r1 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "GoogleVisionAi_Sep2019", m1_opts);

    std::string m2_opts = "sim_user=no_sim_user;model=boolean;model_true_threshold=0.000598001;transform=linear_01;"s;
    auto r2 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "GoogleVisionAi_Sep2019", m2_opts);

    std::string m5_opts =
        "model=vector_space;model_term_tf=natural;model_term_idf=none;model_query_tf=augmented;model_query_idf=idf;model_dist_fn=cosine;model_IDF_method=static_threshold;model_IDF_method_true_threshold=0.001;transform=softmax;"s;
    auto r5 = ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "GoogleVisionAi_Sep2019", m5_opts);
  }
}

/**
 * Tests basic functionality of `W2VV_BoW_Dec2019` data pack methods.
 *
 * \param Reference to ranker instance.
 */
static void test_data_pack__W2VV_BoW_Dec2019(ImageRanker& ranker)
{
  LOGP("===========================================================");
  LOGP("    Running `W2VV_BoW_Dec2019` basic tests");
  LOGP("===========================================================");

  auto r1 = ranker.rank_frames(
      std::vector<std::string>{ "A man wearing a black t-shirt and pants riding a bike on a concrete surface." },
      "ITECTiny_W2VV_BoW_Dec2019", "model=w2vv_bow_plain;", 1000, true, 498);

  auto r2 =
      ranker.rank_frames(std::vector<std::string>{ "An old black car riding in a desert with a few bushes nearby." },
                         "ITECTiny_W2VV_BoW_Dec2019", "model=w2vv_bow_plain;", 1000, true, 123);

  std::string m11_opts = "model=w2vv_bow_plain;"s;
  auto r11 =
      ranker.run_model_test(eUserQueryOrigin::SEMI_EXPERTS, "ITECTiny_W2VV_BoW_Dec2019", m11_opts, true, 100, true);
}

/**
 * Tests basic functionality of `ITECTiny_W2VV_BoW_Dec2019` data pack methods.
 *
 * \param Reference to ranker instance.
 */
static void test_data_pack__ITECTiny_W2VV_BoW_Dec2019(ImageRanker& /*ranker*/)
{
  LOGP("===========================================================");
  LOGP("    Running `W2VV_BoW_Dec2019` basic tests");
  LOGP("===========================================================");
}

/**
 * Tests basic functionality of other \ref ImageRanker methods.
 *
 * \param Reference to ranker instance.
 */
static void test_general_methods(ImageRanker& ranker)
{
  LOGP("===========================================================");
  LOGP("    Running general methods' basic tests");
  LOGP("===========================================================");

  /*
   * Get infos
   */
  {
    auto r1 = ranker.get_loaded_imagesets_info();
    auto r2 = ranker.get_loaded_data_packs_info();
  }
}
}
;

}  // namespace image_ranker