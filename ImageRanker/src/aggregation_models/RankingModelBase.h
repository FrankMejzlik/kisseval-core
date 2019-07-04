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
  virtual void SetModelSettings(AggModelSettings settingsString) = 0;

  virtual std::pair<std::vector<size_t>, size_t> GetRankedImages(
    const std::vector<CnfFormula>& queryFormulae,
    TransformationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::map<size_t, std::unique_ptr<Image>>& _imagesCont,
    size_t numResults = SIZE_T_ERROR_VALUE,
    size_t targetImageId = SIZE_T_ERROR_VALUE
  ) const = 0;

  virtual ChartData RunModelTest(
    TransformationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::vector<std::vector<UserImgQuery>>& testQueries,
    const std::map<size_t, std::unique_ptr<Image>>& _imagesCont
  ) const = 0;
};
