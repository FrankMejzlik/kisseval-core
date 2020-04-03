#pragma once

#include <assert.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "utility.h"

#include "Image.hpp"


#include "KeywordsContainer.h"

#include "data_packs/VIRET_based/transformations/TransformationFunctionBase.h"
#include "data_packs/VIRET_based/ranking_models/RankingModelBase.h"

class RankingModelBase
{



#if 0
  // Required API for ranking models
 public:
    virtual std::pair<std::vector<size_t>, size_t> GetRankedImages(
      const std::vector<std::string>& user_queries, 
      TransformationFunctionBase* pAggregation, const std::vector<float>* pIndexKwFrequency,
      const std::vector<std::unique_ptr<Image>>& _imagesCont,
      const std::map<eVocabularyId, KeywordsContainer>& keywordContainers, size_t numResults = SIZE_T_ERROR_VALUE,
      size_t targetImageId = SIZE_T_ERROR_VALUE) const = 0;

  virtual ChartData RunModelTest(DataId data_ID, TransformationFunctionBase* pAggregation,
                                 const std::vector<float>* pIndexKwFrequency,
                                 const std::vector<std::vector<UserImgQuery>>& testQueries,
                                 const std::vector<std::unique_ptr<Image>>& _imagesCont,
                                 const std::map<eVocabularyId, KeywordsContainer>& keywordContainers) const = 0;

  virtual ChartData RunModelTestWithOrigQueries(
      DataId data_ID, TransformationFunctionBase* pTransformFn, const std::vector<float>* pIndexKwFrequency,
      const std::vector<std::vector<UserImgQuery>>& testQueries,
      const std::vector<std::vector<UserImgQuery>>& testQueriesOrig, const std::vector<std::unique_ptr<Image>>& images,
      const std::map<eVocabularyId, KeywordsContainer>& keywordContainers) const = 0;

#endif
};
