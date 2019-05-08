#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "common.h"
#include "utility.h"

class RankingModelBase
{
  // Required API for ranking models
public:
  virtual void SetModelSettings(ModelSettings settingsString) = 0;
  virtual std::pair<std::vector<size_t>, size_t> GetRankedImages(
    CnfFormula queryFormula,
    AggregationFunctionBase* pAggregation,
    const std::unordered_map<size_t, std::unique_ptr<Image>>& _imagesCont,
    size_t numResults = SIZE_T_ERROR_VALUE,
    size_t targetImageId = SIZE_T_ERROR_VALUE
  ) const = 0;

  virtual ChartData RunModelTest(
    AggregationFunctionBase* pAggregation,
    const std::vector<UserImgQuery>& testQueries,
    const std::unordered_map<size_t, std::unique_ptr<Image>>& _imagesCont
  ) const = 0;
};
