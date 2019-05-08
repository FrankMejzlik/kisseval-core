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
    float m_trueTreshold;
    eQueryOperations m_queryOperation;
  };

public:
  virtual void SetModelSettings(ModelSettings settingsString) override
  {
    // If setting 0 set
    if (settingsString.size() >= 1 && settingsString[0].size() >= 0)
    {
      _settings.m_trueTreshold = strToFloat(settingsString[0]);
    }
    // If setting 1 set
    if (settingsString.size() >= 2 && settingsString[1].size() >= 0)
    {
      _settings.m_queryOperation = static_cast<eQueryOperations>(strToInt(settingsString[1]));
    }
  }

  virtual std::pair<std::vector<size_t>, size_t> GetRankedImages(
    CnfFormula queryFormula,
    size_t aggId,
    const std::unordered_map<size_t, std::unique_ptr<Image>>& _imagesCont,
    size_t numResults,
    size_t targetImageId
  ) override
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
      const std::vector<float>* pImgRankingVector{ pImg->GetAggregationVectorById(aggId) };


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

      float clauseRanking{ 0.0f };

      // Itarate through clauses connected with AND
      for (auto&& clause : queryFormula)
      {
        // Iterate through all variables in clause
        for (auto&& var : clause)
        {
          auto ranking{ (*pImgRankingVector)[var.second] };

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


private:
  Settings GetDefaultSettings() const
  {
    // Return default settings instance
    return Settings{ 0.01f , eQueryOperations::cMultiplyAdd };
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
