#pragma once

#include "RankingModelBase.h"

class ViretModel :
  public RankingModelBase
{
public:
  enum class eQueryOperations
  {
    cMultiplyAdd,
    cAddOnly
  };

  struct Settings
  {
    unsigned int m_keywordFrequencyHandling;
    float m_trueTreshold;
    eQueryOperations m_queryOperation;
  };

public:
  virtual void SetModelSettings(ModelSettings settingsString) override
  {
    _settings = GetDefaultSettings();

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
      _settings.m_queryOperation = static_cast<eQueryOperations>(strToInt(settingsString[2]));
    }
  }

  virtual std::pair<std::vector<size_t>, size_t> GetRankedImages(
    CnfFormula queryFormula,
    AggregationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::unordered_map<size_t, std::unique_ptr<Image>>& _imagesCont,
    size_t numResults,
    size_t targetImageId
  ) const  override
  {
    // If want all results
    if (numResults == SIZE_T_ERROR_VALUE)
    {
      numResults = _imagesCont.size();
    }

    // Comparator lambda for priority queue
    auto cmp = [](const std::pair<float, size_t>& left, const std::pair<float, size_t>& right)
    {
      return left.first < right.first;
    };

    // Reserve enough space in container
    std::vector<std::pair<float, size_t>> container;
    container.reserve(_imagesCont.size());

    std::priority_queue<
      std::pair<float, size_t>, 
      std::vector<std::pair<float, size_t>>, 
      decltype(cmp)> maxHeap(cmp, std::move(container));

    
    std::pair<std::vector<size_t>, size_t> result;
    result.first.reserve(numResults);

    // Check every image if satisfies query formula
    for (auto&& [imgId, pImg] : _imagesCont)
    {
      // Prepare pointer for ranking vector aggregation data
      const std::vector<float>* pImgRankingVector{ pImg->GetAggregationVectorById(pAggregation->GetGuidFromSettings()) };


      // If Add and Multiply
      float imageRanking{ 1.0f };

      // Choose correct operation
      switch (_settings.m_queryOperation)
      {
      case eQueryOperations::cMultiplyAdd:
        imageRanking = 1.0f;
        break;

      case eQueryOperations::cAddOnly:
        imageRanking = 0.0f;
        break;

      default:
        LOG_ERROR("Unknown query operation.");
      }

      
      float numNegate{0.0f};
      // Itarate through clauses connected with AND
      for (auto&& clause : queryFormula)
      {
        float clauseRanking{ 0.0f };

        // Iterate through all variables in clause
        for (auto&& var : clause)
        {
          auto ranking{ (*pImgRankingVector)[var.second] };

          float factor{1.0f};
          // Choose correct KF handling
          switch (_settings.m_keywordFrequencyHandling)
          {
            // No care about TFIDF
          case 0:
            
            break;

            // Multiply with TFIDF
          case 1:
            factor =  (*pIndexKwFrequency)[var.second];
            ranking = ranking * factor;
            break;

          default:
            LOG_ERROR("Unknown keyword freq operation.");
          }


          // Is negative
          if (var.first)
          {
            numNegate += ranking;
            continue;
          }

          // Skip all labels with too low ranking
          if (ranking < _settings.m_trueTreshold)
          {
            continue;
          }

          // Add up labels in one clause
          clauseRanking += ranking;
        }

        // Choose correct operation
        switch (_settings.m_queryOperation)
        {
        case eQueryOperations::cMultiplyAdd:
          imageRanking = imageRanking * clauseRanking;
          break;

        case eQueryOperations::cAddOnly:
          imageRanking = imageRanking + clauseRanking;
          break;

        default:
          LOG_ERROR("Unknown query operation.");
        }

      }

      if (numNegate > 0) 
      {
        imageRanking /= ((numNegate* 1000 * queryFormula.size()) + 1);
      }

      // Insert result to max heap
      maxHeap.push(std::pair(imageRanking, pImg->m_imageId));
    }

    size_t sizeHeap{ maxHeap.size() };

    // Get out sorted results
    for (size_t i = 0ULL; i < sizeHeap; ++i)
    {
      auto pair = maxHeap.top();
      maxHeap.pop();

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
    AggregationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::vector<UserImgQuery>& testQueries,
    const std::unordered_map<size_t, std::unique_ptr<Image>>& _imagesCont
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
    for (auto&& [imgId, queryFormula] : testQueries)
    {
      auto resultImages = GetRankedImages(queryFormula, pAggregation, pIndexKwFrequency, _imagesCont, 0ULL, imgId);

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
  Settings GetDefaultSettings() const
  {
    // Return default settings instance
    return Settings{ 0U, 0.01f , eQueryOperations::cMultiplyAdd };
  }

private:
  Settings _settings;

public:
  static float m_trueTresholdFrom;
  static float m_trueTresholdTo;
  static float m_trueTresholdStep;
  static std::vector<float> m_trueTresholds;

  static std::vector<uint8_t> m_queryOperations;
};
