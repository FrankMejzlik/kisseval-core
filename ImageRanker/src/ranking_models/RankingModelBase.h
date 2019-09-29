#pragma once

#include <assert.h>

#include <string>
#include <vector>
#include <algorithm>
#include <map>

#include "common.h"
#include "utility.h"

#include "Image.hpp"
#include "TransformationFunctionBase.h"

#include "KeywordsContainer.h"


class RankingModelBase
{
  // Required API for ranking models
public:
  virtual void SetModelSettings(RankingModelSettings settingsString) = 0;

  virtual std::pair<std::vector<size_t>, size_t> GetRankedImages(
    const std::vector<CnfFormula>& queryFormulae,
    KwScoringDataId kwScDataId,
    TransformationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::vector<std::unique_ptr<Image>>& _imagesCont,
    const std::map<eKeywordsDataType, KeywordsContainer>& keywordContainers,
    size_t numResults = SIZE_T_ERROR_VALUE,
    size_t targetImageId = SIZE_T_ERROR_VALUE
  ) const = 0;

  virtual ChartData RunModelTest(
    KwScoringDataId kwScDataId,
    TransformationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::vector<std::vector<UserImgQuery>>& testQueries,
    const std::vector<std::unique_ptr<Image>>& _imagesCont,
    const std::map<eKeywordsDataType, KeywordsContainer>& keywordContainers
  ) const = 0;
};
