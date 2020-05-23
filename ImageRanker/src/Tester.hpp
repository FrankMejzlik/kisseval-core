#pragma once

#include "common.h"
#include "utility.h"

#include "ImageRanker.h"

using namespace image_ranker;

namespace image_ranker
{
class Tester
{
  static bool test_public_methods(ImageRanker& ranker, const ImageRanker::Config& cfg)
  {
    for (auto&& pack : cfg.VIRET_packs)
    {
      if (pack.ID == "NasNet2019")
      {
      }
      else if (pack.ID == "GoogLeNet2019")
      {
      }
      else if (pack.ID == "ITECTiny_NasNet2019")
      {
      }
    }

    for (auto&& pack : cfg.Google_packs)
    {
      if (pack.ID == "GoogleVisionAi_Sep2019")
      {
      }
    }

    for (auto&& pack : cfg.W2VV_packs)
    {
      if (pack.ID == "W2VV_BoW_Dec2019")
      {
      }
      else if (pack.ID == "ITECTiny_W2VV_BoW_Dec2019")
      {
      }
    }
  }

  static void grid_test_find_true_thresholds(ImageRanker& ranker)
  {
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
        auto res = ranker.run_model_test(
            eUserQueryOrigin::SEMI_EXPERTS, "NasNet2019",
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
};
}  // namespace image_ranker