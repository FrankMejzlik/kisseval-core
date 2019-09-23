#pragma once

#include "RankingModelBase.h"

#include <queue>

class BooleanBucketModel :
  public RankingModelBase
{
public:
  enum class eInBucketSorting
  {
    eNone,
    eSum,
    eMax
  };

  struct Settings
  {
    unsigned int m_keywordFrequencyHandling;
    float m_trueTreshold;
    eInBucketSorting m_inBucketSorting;

    eTempQueryOpOutter m_tempQueryOutterOperation;
    eTempQueryOpInner m_tempQueryInnerOperation;
  };


  // Methods
public:
  Settings GetDefaultSettings() const
  {
    // Return default settings instance
    return Settings{ 
      0U, 
      0.01f, 
      eInBucketSorting::eNone,
      eTempQueryOpOutter::cProduct, // Temporal query OUTTER operation
      eTempQueryOpInner::cMax // Temporal query INNER operation 
    };
  }

  virtual void SetModelSettings(RankingModelSettings settingsString) override
  {
    _settings =  GetDefaultSettings();

    // If setting 0 set
    if (settingsString.size() >= 1 && settingsString[0].size() >= 0)
    {
      _settings.m_keywordFrequencyHandling = static_cast<unsigned int>(strToInt(settingsString[0]));
    }
    // If setting 1 set
    if (settingsString.size() >= 2 && settingsString[1].size() >= 0)
    {
      _settings.m_trueTreshold = strToFloat(settingsString[1]);
    }
    // If setting 2 set
    if (settingsString.size() >= 3 && settingsString[2].size() >= 0)
    {
      _settings.m_inBucketSorting = static_cast<eInBucketSorting>(strToInt(settingsString[2]));
    }

    // If setting 3 set
    if (settingsString.size() >= 4 && settingsString[3].size() >= 0)
    {
      _settings.m_tempQueryOutterOperation = static_cast<eTempQueryOpOutter>(strToInt(settingsString[3]));
    }
    // If setting 4 set
    if (settingsString.size() >= 5 && settingsString[4].size() >= 0)
    {
      _settings.m_tempQueryInnerOperation = static_cast<eTempQueryOpInner>(strToInt(settingsString[4]));
    }

  }

  virtual std::pair<std::vector<size_t>, size_t> GetRankedImages(
    const std::vector<CnfFormula>& queryFormulae,
    KwScoringDataId kwScDataId,
    TransformationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::vector<std::unique_ptr<Image>>& _imagesCont,
    size_t numResults = SIZE_T_ERROR_VALUE,
    size_t targetImageId = SIZE_T_ERROR_VALUE
  ) const override
  {
    // If want all results
    if (numResults == SIZE_T_ERROR_VALUE)
    {
      numResults = _imagesCont.size();
    }

    auto cmp = [](const std::pair<std::pair<size_t, float>, size_t>& left, const std::pair<std::pair<size_t, float>, size_t>& right)
    {
      if (left.first.first > right.first.first)
      {
        // First is greater
        return true;
      }
      else if (left.first.first == right.first.first)
      {
        if (left.first.second < right.first.second)
        {
          return true;
        }
        else
        {
          return false;
        }
      }
      else
      {
        return false;
      }
    };

    // Reserve enough space in container
    std::vector<std::pair<std::pair<size_t, float>, size_t>> container;
    container.reserve(_imagesCont.size());

    std::priority_queue<
      std::pair<std::pair<size_t, float>, size_t>, 
      std::vector<std::pair<std::pair<size_t, float>, size_t>>, 
      decltype(cmp)
    > minHeap(cmp, std::move(container));

    // Extract desired number of images out of min heap
    std::pair<std::vector<size_t>, size_t> result;
    result.first.reserve(numResults);


    // Check every image if satisfies query formula
    for (auto&& pImg : _imagesCont)
    {
      // Prepare pointer for ranking vector aggregation data
      const std::vector<float>* pImgRankingVector{ pImg->GetAggregationVectorById(kwScDataId, pAggregation->GetGuidFromSettings()) };

      size_t imageSucc{ 0ULL };
      float imageSubRank{ 0.0f };

      // Itarate through clauses connected with AND
      for (auto&& clause : queryFormulae[0])
      {
        bool clauseSucc{ false };
        size_t negatePenalty{0ULL};

        // Iterate through predicates
        for (auto&& var : clause)
        {
          float factor{ 1.0f };
          auto binRankVal{ (*pImgRankingVector)[var.second] };

          switch (_settings.m_keywordFrequencyHandling)
          {
            // No care about TFIDF
          case 0:

            break;

            // Multiply with TFIDF
          case 1:
            factor = (*pIndexKwFrequency)[var.second];
            binRankVal = binRankVal * factor;
            break;

          default:
            LOG_ERROR("Unknown keyword freq operation.");
          }


          // If this variable satisfies this clause
          if (binRankVal >= _settings.m_trueTreshold)
          {
            clauseSucc = true;

            // If is negate
            if (var.first)
            {
              negatePenalty = 1;
            }
            break;
          }

          {
            // If is negate
            if (var.first)
            {
              binRankVal = -binRankVal;
            }

            switch (_settings.m_inBucketSorting)
            {
            case eInBucketSorting::eNone:
            {
              // No sorting within bucket
            }
            break;

            case eInBucketSorting::eSum:
            {
              // Summ sort
              imageSubRank += binRankVal;
            }
            break;

            case eInBucketSorting::eMax:
            {
              // Get max
              if (imageSubRank < binRankVal)
              {
                imageSubRank = binRankVal;
              }
            }
            break;

            default:
              LOG_ERROR("Unknown query operation.");
            }
          }
        }

        // If this clause not satisfied
        if (!clauseSucc)
        {
          ++imageSucc;
        }
      }

      // Insert result to min heap
      minHeap.push(std::pair(std::pair(imageSucc, imageSubRank), pImg->m_imageId));
    }

    size_t sizeHeap{ minHeap.size() };
    for (size_t i = 0ULL; i < sizeHeap; ++i)
    {
      auto pair = minHeap.top();
      minHeap.pop();

      // If is target image, save it
      if (targetImageId == pair.second)
      {
        result.second = i + 1;
      }

      if (i < numResults)
      {
        result.first.emplace_back(pair.second);
      }

    }

    return result;
  }


  virtual ChartData RunModelTest(
    KwScoringDataId kwScDataId,
    TransformationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::vector<std::vector<UserImgQuery>>& testQueries,
    const std::vector<std::unique_ptr<Image>>& _imagesCont
  ) const override
  {
    uint32_t maxRank = (uint32_t)_imagesCont.size();

    // To have 100 samples
    uint32_t scaleDownFactor = maxRank / CHART_DENSITY;

    std::vector<std::pair<uint32_t, uint32_t>> result;
    result.resize(CHART_DENSITY + 1);

    uint32_t label{ 0ULL };
    for (auto&& column : result)
    {
      column.first = label;
      label += scaleDownFactor;
    }

    // Iterate over test queries
    for (auto&& singleQuery : testQueries)
    {
      auto imgId = std::get<0>(singleQuery[0]);

      std::vector<CnfFormula> formulae;
      for (auto&&[imgId, queryFormula, withExamples] : singleQuery)
      {
        formulae.push_back(queryFormula);
      }

      auto resultImages = GetRankedImages(formulae, kwScDataId, pAggregation, pIndexKwFrequency, _imagesCont, 0ULL, imgId);

      size_t transformedRank = resultImages.second / scaleDownFactor;

      // Increment this hit
      ++result[transformedRank].second;
    }


    uint32_t currCount{ 0ULL };

    // Compute final chart values
    for (auto&& r : result)
    {
      uint32_t tmp{ r.second };
      r.second = currCount;
      currCount += tmp;
    }

    return result;
  }

private:
  Settings _settings;

public:
  static float m_trueTresholdFrom;
  static float m_trueTresholdTo;
  static float m_trueTresholdStep;
  static std::vector<float> m_trueTresholds;

  static std::vector<uint8_t> m_inBucketOrders;
};