#pragma once

#include "RankingModelBase.h"

#include <queue>

class ViretModel :
  public RankingModelBase
{
public:
  /*!
   * List of possible settings what how to calculate rank 
   */
  enum class eQueryOperations
  {
    cMultSum = 0,
    cMultMax = 1,
    cSumSum = 2,
    cSumMax = 3,
    cMaxMax= 4
  };

  struct Settings
  {
    //! How to handle keyword frequency in ranking
    unsigned int m_keywordFrequencyHandling;

    //! What values are considered significant enough to calculate with them
    float m_trueTreshold;

    //! What operations are executed when creating rank for given image
    eQueryOperations m_queryOperation;

    eTempQueryOpOutter m_tempQueryOutterOperation;
    eTempQueryOpInner m_tempQueryInnerOperation;
  };

public:
  virtual void SetModelSettings(RankingModelSettings settingsString) override
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


  float GetImageQueryRanking(
    KwScoringDataId kwScDataId,
    Image* pImg,
    const std::vector<CnfFormula>& queryFormulae,
    TransformationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::vector<std::unique_ptr<Image>>& _imagesCont
  ) const
  {
#if LOG_DEBUG_IMAGE_RANKING 
    std::cout << "IMAGE ID  " << std::to_string(imgId) << std::endl;
    std::cout << "======================" << std::endl;
#endif

    // Prepare pointer for ranking vector aggregation data
    const std::vector<float>* pImgRankingVector{ pImg->GetAggregationVectorById(kwScDataId, pAggregation->GetGuidFromSettings()) };

#if LOG_DEBUG_IMAGE_RANKING 
    std::cout << "Precomputed vector: ";
    {
      size_t i{ 0_z };
      for (auto&& bin : *pImgRankingVector)
      {
        std::cout << "(" << std::to_string(i) << ", " << std::to_string(bin) << "),";

        ++i;
      }
    }
    std::cout << std::endl;
#endif

    // Initialize this image ranking value
    float imageRanking{ 1.0f };

    // ========================================
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    // SETTINGS: Choose correct operation with ranking
    switch (_settings.m_queryOperation)
    {
    case eQueryOperations::cMultMax:
    case eQueryOperations::cMultSum:
      imageRanking = 1.0f;
      break;

    case eQueryOperations::cSumSum:
    case eQueryOperations::cSumMax:
    case eQueryOperations::cMaxMax:
      imageRanking = 0.0f;
      break;

    default:
      LOG_ERROR("Unknown query operation "s + std::to_string(static_cast<int>(_settings.m_queryOperation)) + "."s);
    }
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ========================================

    float negateFactor{ 0.0f };


#if LOG_DEBUG_IMAGE_RANKING 
    std::cout << "\n\nSTART => imageRanking = " << imageRanking << std::endl;

    size_t clauseCounter{ 0_z };
#endif


    // Itarate through clauses connected with AND
    for (auto&& clause : queryFormulae[0])
    {
#if LOG_DEBUG_IMAGE_RANKING 
      std::cout << "==== new clause ====" << std::endl;
      std::cout << "Processing clause " << std::to_string(clauseCounter) << std::endl;
#endif
      float clauseRanking{ 0.0f };

      // Iterate through all variables in clause
      for (auto&& literal : clause)
      {
        auto currKwRanking{ (*pImgRankingVector)[literal.second] };

#if LOG_DEBUG_IMAGE_RANKING 
        std::cout << "\t==== new literal ====" << std::endl;
        std::cout << "\tbinIndex = " << std::to_string(literal.second) << std::endl;
        std::cout << "\tclauseRanking = " << std::to_string(clauseRanking) << std::endl;
        std::cout << "\thisKeywordRanking = vector[binIndex] = " << std::to_string(currKwRanking) << std::endl;
        std::cout << "\---" << std::endl;
#endif

        float factor{ 1.0f };

        // ========================================
        // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        // SETTINGS: Choose correct KF handling
        switch (_settings.m_keywordFrequencyHandling)
        {
          // No care about TFIDF
        case 0:

          break;

          // Multiply with TFIDF
        case 1:
          factor = (*pIndexKwFrequency)[literal.second];
          currKwRanking = currKwRanking * factor;
          break;

        default:
          LOG_ERROR("Unknown keyword freq operation.");
        }
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ========================================


        // Is negative
        if (literal.first)
        {
          negateFactor += currKwRanking;
          continue;
        }

        // Skip all labels with too low ranking
        if (currKwRanking < _settings.m_trueTreshold)
        {
          continue;
        }

        // ========================================
        // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        // SETTINGS: Choose what to do with caluse ranks
        switch (_settings.m_queryOperation)
        {
          // For sum based
        case eQueryOperations::cMultSum:
        case eQueryOperations::cSumSum:
        {
          // just accumulate
          clauseRanking += currKwRanking;
        }
        break;

        // For max based
        case eQueryOperations::cMultMax:
        case eQueryOperations::cSumMax:
        case eQueryOperations::cMaxMax:
        {
#if LOG_DEBUG_IMAGE_RANKING 
          std::cout << "\tclauseRanking = std::max(" << std::to_string(clauseRanking) << ", " << std::to_string(currKwRanking) << ")" << std::endl;
#endif

          // Get just maximum
          clauseRanking = std::max(clauseRanking, currKwRanking);


#if LOG_DEBUG_IMAGE_RANKING 
          std::cout << "\tclauseRanking = " << std::to_string(clauseRanking) << std::endl << std::endl;
          std::cout << "\t==== literal ends ====" << std::endl;
#endif
        }
        break;

        default:
          LOG_ERROR("Unknown query operation "s + std::to_string(static_cast<int>(_settings.m_queryOperation)) + "."s);
        }
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ========================================
      }

      // ========================================
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      // SETTINGS: Choose correct operation
      switch (_settings.m_queryOperation)
      {
        // Outter operatin Multiplication
      case eQueryOperations::cMultSum:
      case eQueryOperations::cMultMax:
      {
        // Multiply all clause rankings
        imageRanking = imageRanking * clauseRanking;
      }
      break;

      // Outter operatin Sum
      case eQueryOperations::cSumSum:
      case eQueryOperations::cSumMax:
      {

#if LOG_DEBUG_IMAGE_RANKING 
        std::cout << "imageRanking = " << std::to_string(imageRanking) << " + " << std::to_string(clauseRanking) << std::endl;
#endif

        // Add all clause rankings
        imageRanking = imageRanking + clauseRanking;

#if LOG_DEBUG_IMAGE_RANKING 
        std::cout << "imageRanking = " << std::to_string(imageRanking) << std::endl;
        std::cout << "==== clause ends ====" << std::endl;
#endif
      }
      break;

      // Outter operatin Max
      case eQueryOperations::cMaxMax:
      {
        // Get just maximum
        imageRanking = std::max(imageRanking, clauseRanking);
      }
      break;

      default:
        LOG_ERROR("Unknown query operation "s + std::to_string(static_cast<int>(_settings.m_queryOperation)) + "."s);
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ========================================

#if LOG_DEBUG_IMAGE_RANKING
      ++clauseCounter;
#endif
    }


    // If there were some negative keywords
    if (negateFactor > 0)
    {
      imageRanking /= ((negateFactor * queryFormulae[0].size()) + 1);
    }

    return imageRanking;
  }

  float GetImageTemporalQueryRanking(
    KwScoringDataId kwScDataId,
    Image* pImg,
    const std::vector<CnfFormula>& queryFormulae,
    TransformationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::vector<std::unique_ptr<Image>>& _imagesCont
  ) const
  {
    //
    // Initialize subranking calculations
    //

    // Prepare final ranking variable
    float resultRanking;
    switch (_settings.m_tempQueryInnerOperation)
    {
    case eTempQueryOpInner::cProduct:
      resultRanking = 1.0f;
      break;

    case eTempQueryOpInner::cMax:
    case eTempQueryOpInner::cSum:
      resultRanking = 0.0f;
      break;

    default:
      LOG_ERROR("Uknown temporal inner operation.");
      break;
    }

    // Get iterator to this image
    auto imgIt{ _imagesCont.begin() + pImg->m_index };

    // If item not found
    if (imgIt == _imagesCont.end())
    {
      LOG_ERROR("Image not found in map.");
    }


    //
    // Iterate through all successors and compute their rankings
    //

    size_t count{ 0_z };
    // If this frame has some succesor frames
    while (pImg->m_numSuccessorFrames > 0)
    {
      if (count >= MAX_SUCC_CHECK_COUNT)
      {
        break;
      }

      auto bckpImgIt = imgIt;

      // Move iterator to successor
      ++imgIt;

      assert(imgIt != _imagesCont.end());

      // Get image ranking of image
      auto imageRanking{ GetImageQueryRanking(
        kwScDataId,
        imgIt->get(),
        queryFormulae, pAggregation, pIndexKwFrequency, _imagesCont
      )};


      // Use correct outter temporal query operation
      switch (_settings.m_tempQueryInnerOperation)
      {
      case eTempQueryOpInner::cProduct:
        resultRanking = resultRanking * imageRanking;
        break;

      case eTempQueryOpInner::cMax:
        resultRanking = std::max(resultRanking, imageRanking);
        break;

      case eTempQueryOpInner::cSum:
        resultRanking = resultRanking * imageRanking;
        break;

      default:
        LOG_ERROR("Uknown temporal inner operation.");
      }  

      ++count;
    }

    return resultRanking;
  }

  virtual std::pair<std::vector<size_t>, size_t> GetRankedImages(
    const std::vector<CnfFormula>& queryFormulae,
    KwScoringDataId kwScDataId,
    TransformationFunctionBase* pAggregation,
    const std::vector<float>* pIndexKwFrequency,
    const std::vector<std::unique_ptr<Image>>& _imagesCont,
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
    for (auto&& pImg : _imagesCont)
    {
      //
      // Get ranking of main frame
      //

      // Get image ranking of main image
      auto imageRanking{ GetImageQueryRanking(
        kwScDataId,
        pImg.get(),
        queryFormulae, pAggregation, pIndexKwFrequency, _imagesCont
      )};


      //
      // Count in temporal query if present
      //

      // If temporal query provided
      if (queryFormulae.size() > 1_z)
      {
        std::vector<CnfFormula> subFormulae;
        subFormulae.push_back(queryFormulae[1]);

        auto tempQueryRanking{ GetImageTemporalQueryRanking(
          kwScDataId,
          pImg.get(),
          subFormulae, pAggregation, pIndexKwFrequency, _imagesCont
        )};


        // Use correct outter temporal query operation
        switch (_settings.m_tempQueryOutterOperation)
        {
        case eTempQueryOpOutter::cProduct:

          imageRanking = imageRanking * tempQueryRanking;

          break;

        case eTempQueryOpOutter::cSum:

          imageRanking = imageRanking + tempQueryRanking;

          break;

        default:
          LOG_ERROR("Uknown temporal outter operation.");
        }
      }

      //
      // Finalize ranking
      // 

#if LOG_DEBUG_IMAGE_RANKING
      std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
      std::cout << "!!!! FINAL RANK = " << std::to_string(imageRanking) << std::endl;
      std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
#endif

      // Insert result to max heap
      maxHeap.push(std::pair(imageRanking, pImg->m_imageId));
    }


    /*
     * Ranking computed and stored in max heap,
     * now we'll just extract it to desired result structure
     */
     
    // Get size of heap
    size_t sizeHeap{ maxHeap.size() };

    // Iterate over popping items in heap
    for (size_t i{ 0_z }; i < sizeHeap; ++i)
    {
      // Get copy of the greatest element from heap
      auto pair = maxHeap.top();
      maxHeap.pop();

      // If this is target image
      if (targetImageId == pair.second)
      {
        // Store its rank (rank is one greater than index)
        result.second = i + 1;
      }

      // If we extracted enough items user asked for
      if (i < numResults)
      {
        // PLace it inside result structure
        result.first.emplace_back(std::move(pair.second));
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
  Settings GetDefaultSettings() const
  {
    // Return default settings instance
    return Settings{ 
      0U, // Keyword frequency handling
      0.01f, // Ignore threshold
      eQueryOperations::cMultSum, // Normal query operations
      eTempQueryOpOutter::cProduct, // Temporal query OUTTER operation
      eTempQueryOpInner::cMax // Temporal query INNER operation 
    }; 
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
