#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
using namespace std::string_literals;

#include "common.h"
#include "utility.h"

class GridTest
{
 public:
  static std::atomic<size_t> numCompletedTests;

  static void ProgressCallback() { GridTest::numCompletedTests.operator++(); }

  static std::pair<uint8_t, uint8_t> GetGridTestProgress()
  {
    return std::pair((uint8_t)GridTest::numCompletedTests, (uint8_t)m_testSettings.size());
  }

  static void ReportTestProgress()
  {
    while (true)
    {
      if (numCompletedTests >= m_testSettings.size())
      {
        break;
      }

      LOG("Test progress is "s + std::to_string(GridTest::numCompletedTests) + "/"s +
          std::to_string(m_testSettings.size()));

      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
  }

  static std::vector<InputDataTransformId> m_aggregations;
  static std::vector<UserDataSourceId> m_queryOrigins;
  static std::vector<RankingModelId> m_rankingModels;
  static std::vector<TestSettings> m_testSettings;
};
