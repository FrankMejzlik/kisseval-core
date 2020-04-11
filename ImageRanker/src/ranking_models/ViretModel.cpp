
#include "ViretModel.h"

ViretModel::Options ViretModel::ParseOptionsString(const std::string& options_string) { return ViretModel::Options(); }

std::vector<FrameId> ViretModel::rank_frames(const Matrix<float>& data_mat, const KeywordsContainer& keywords,
                                             const std::vector<std::string>& user_query,
                                             const std::string& options) const
{
#if 0
#if LOG_DEBUG_IMAGE_RANKING

    auto kwsId{std::get<0>(data_ID)};
    auto& kwCont{keywordContainers.at(kwsId)};

    std::cout << "Image ID: " << std::to_string(pImg->m_imageId) << ": " << pImg->m_filename << std::endl;
    std::cout << "======================" << std::endl;
#endif

    // Prepare pointer for ranking vector aggregation data
    const std::vector<float>* pImgRankingVector{
        pImg->GetAggregationVectorById(data_ID, pTransformFn->GetGuidFromSettings())};

#if LOG_DEBUG_IMAGE_RANKING

    std::cout << "Precomputed data: " << std::endl;
    {
      size_t i{0_z};
      for (auto&& bin : *pImgRankingVector)
      {
        if (bin > GOOGLE_AI_NO_LABEL_SCORE)
        {
          auto pKw{kwCont.GetKeywordConstPtrByVectorIndex(i)};

          std::cout << pKw->m_wordnetId << ": " << pKw->m_word << " -> " << std::to_string(bin) << std::endl;
        }

        ++i;
      }
    }
    std::cout << "======================" << std::endl;
    std::cout << std::endl;
#endif

    // Initialize this image ranking value
    float imageRanking{1.0f};

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

    float negateFactor{0.0f};

#if LOG_DEBUG_IMAGE_RANKING
    std::cout << "queryFormula = " << std::endl;
    for (auto&& clause : queryFormulae[0])
    {
      std::cout << " ( ";
      for (auto&& literal : clause)
      {
        auto currKwRanking{(*pImgRankingVector)[literal.second]};
        auto pKw{kwCont.GetKeywordConstPtrByVectorIndex(literal.second)};

        std::cout << pKw->m_word << "<" << currKwRanking << ">"
                  << " + ";
      }
      std::cout << " )  * " << std::endl;
    }

    std::cout << "====================================" << std::endl << std::endl;
    std::cout << "=> => => START COMPUTE IMAGE SCORE" << std::endl;
    std::cout << "imageRanking = " << imageRanking << std::endl;

    size_t clauseCounter{0_z};
#endif

    // Itarate through clauses connected with AND
    for (auto&& clause : queryFormulae[0])
    {
      float clauseRanking{0.0f};

      // Iterate through all variables in clause
      for (auto&& literal : clause)
      {
        auto currKwRanking{(*pImgRankingVector)[literal.second]};

#if LOG_DEBUG_IMAGE_RANKING
        auto pKw{kwCont.GetKeywordConstPtrByVectorIndex(literal.second)};

        std::cout << "\t => " << pKw->m_word << "< " << std::to_string(currKwRanking) << " >" << std::endl;
#endif

        float factor{1.0f};

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
#if LOG_DEBUG_IMAGE_RANKING
            std::cout << "\t < clauseRanking += currKwRanking >" << std::endl;
            std::cout << "\t clauseRanking += " << currKwRanking << std::endl;
#endif
            // just accumulate
            clauseRanking += currKwRanking;

#if LOG_DEBUG_IMAGE_RANKING
            std::cout << "\t clauseRanking = " << clauseRanking << std::endl;
#endif
          }
          break;

          // For max based
          case eQueryOperations::cMultMax:
          case eQueryOperations::cSumMax:
          case eQueryOperations::cMaxMax:
          {
#if LOG_DEBUG_IMAGE_RANKING
            std::cout << "\t < clauseRanking = std::max(clauseRanking, currKwRanking) >" << std::endl;
            std::cout << "\t clauseRanking = std::max(" << clauseRanking << ", " << currKwRanking << ")" << std::endl;
#endif

            // Get just maximum
            clauseRanking = std::max(clauseRanking, currKwRanking);

#if LOG_DEBUG_IMAGE_RANKING
            std::cout << "\t clauseRanking = " << clauseRanking << std::endl;
#endif
          }
          break;

          default:
            LOG_ERROR("Unknown query operation "s + std::to_string(static_cast<int>(_settings.m_queryOperation)) +
                      "."s);
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
#if LOG_DEBUG_IMAGE_RANKING
          std::cout << "< imageRanking = imageRanking * clauseRanking; >" << std::endl;
          std::cout << "imageRanking = " << std::to_string(imageRanking) << " * " << std::to_string(clauseRanking)
                    << std::endl;
#endif
          // Multiply all clause rankings
          imageRanking = imageRanking * clauseRanking;

#if LOG_DEBUG_IMAGE_RANKING
          std::cout << "imageRanking = " << std::to_string(imageRanking) << std::endl;
#endif
        }
        break;

        // Outter operatin Sum
        case eQueryOperations::cSumSum:
        case eQueryOperations::cSumMax:
        {
#if LOG_DEBUG_IMAGE_RANKING
          std::cout << "< imageRanking = imageRanking + clauseRanking; >" << std::endl;
          std::cout << "imageRanking = " << std::to_string(imageRanking) << " + " << std::to_string(clauseRanking)
                    << std::endl;
#endif

          // Add all clause rankings
          imageRanking = imageRanking + clauseRanking;

#if LOG_DEBUG_IMAGE_RANKING
          std::cout << "imageRanking = " << std::to_string(imageRanking) << std::endl;
#endif
        }
        break;

        // Outter operatin Max
        case eQueryOperations::cMaxMax:
        {
#if LOG_DEBUG_IMAGE_RANKING
          std::cout << "< imageRanking = std::max(imageRanking, clauseRanking); >" << std::endl;
          std::cout << " imageRanking = std::max(" << std::to_string(imageRanking) << " + "
                    << std::to_string(clauseRanking) << ")" << std::endl;
#endif

          // Get just maximum
          imageRanking = std::max(imageRanking, clauseRanking);

#if LOG_DEBUG_IMAGE_RANKING
          std::cout << "imageRanking = " << std::to_string(imageRanking) << std::endl;
#endif
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

#if LOG_DEBUG_IMAGE_RANKING
    std::cout << "imageRanking = " << imageRanking << std::endl;
    std::cout << "<= <= <= END COMPUTE IMAGE SCORE" << std::endl;
    std::cout << "====================================" << std::endl << std::endl;
#endif
#endif
  return std::vector<FrameId>();
}

std::vector<FrameId> ViretModel::run_test(
    const Matrix<float>& data_mat, const KeywordsContainer& keywords,
    const std::vector<std::pair<std::vector<std::string>, FrameId>>& test_user_queries, const std::string& options,
    size_t result_points) const
{
  return std::vector<FrameId>();
}

#if 0

  
  

  virtual ChartData RunModelTestWithOrigQueries(
      DataId data_ID, TransformationFunctionBase* pTransformFn, const std::vector<float>* pIndexKwFrequency,
      const std::vector<std::vector<UserImgQuery>>& testQueries,
      const std::vector<std::vector<UserImgQuery>>& testQueriesOrig, const std::vector<std::unique_ptr<Image>>& images,
      const std::map<eVocabularyId, KeywordsContainer>& keywordContainers) const override
  {
    std::vector<size_t> ranks;
#if LOG_DEBUG_RUN_TESTS
    std::cout << "Running model test ... " << std::endl;
    std::cout << "Result chart will have " << MODEL_TEST_CHART_NUM_X_POINTS << " discrete points on X axis"
              << std::endl;
    std::cout << "=====================================" << std::endl;
#endif

#if LOG_PRE_AND_PPOST_EXP_RANKS
    long long int totalRankMove{0};
    long long int currDelta{0};

    size_t origRank{0_z};
    size_t expRank{0_z};
#endif

    uint32_t maxRank = (uint32_t)images.size();

    // To have 100 samples
    uint32_t scaleDownFactor = maxRank / MODEL_TEST_CHART_NUM_X_POINTS;

    std::vector<std::pair<uint32_t, uint32_t>> result;
    result.resize(MODEL_TEST_CHART_NUM_X_POINTS + 1);

    uint32_t label{0ULL};
    for (auto&& column : result)
    {
      column.first = label;
      label += scaleDownFactor;
    }

    size_t iii{0_z};

    // Iterate over test queries
    for (auto&& singleQuery : testQueries)
    {
#if LOG_PRE_AND_PPOST_EXP_RANKS
      {
        auto singleQueryOrig{testQueriesOrig[iii]};
        auto imgId = std::get<0>(singleQueryOrig[0]);

        std::vector<CnfFormula> formulae;
        for (auto&& [imgId, queryFormula, withExamples] : singleQueryOrig)
        {
          formulae.push_back(queryFormula);
        }

        auto resultImages = GetRankedImages(formulae, data_ID, pTransformFn, pIndexKwFrequency, images,
                                            keywordContainers, 0ULL, imgId);

        origRank = resultImages.second;

        std::cout << "-----" << std::endl;
        std::cout << origRank << " => ";
      }
#endif
      auto imgId = std::get<0>(singleQuery[0]);

      std::vector<CnfFormula> formulae;
      for (auto&& [imgId, queryFormula, withExamples] : singleQuery)
      {
        formulae.push_back(queryFormula);
      }

      auto resultImages = GetRankedImages(formulae, data_ID, pTransformFn, pIndexKwFrequency, images,
                                          keywordContainers, 0ULL, imgId);

      // Rank index is -1 from rank
      size_t transformedRank = (resultImages.second - 1) / scaleDownFactor;

      ranks.emplace_back(resultImages.second);

      // Increment this hit
      ++result[transformedRank].second;

#if LOG_PRE_AND_PPOST_EXP_RANKS
      expRank = resultImages.second;
      currDelta = expRank - origRank;

      totalRankMove += currDelta;

      std::cout << expRank << " delta = " << currDelta << ", totalRankDelta = " << totalRankMove << std::endl;

#endif

#if LOG_DEBUG_RUN_TESTS
      std::cout << "----------------------------" << std::endl;
      std::cout << "Image ID " << imgId << "=> " << (resultImages.second - 1) << "/" << maxRank << std::endl;

      size_t from{transformedRank * scaleDownFactor};
      size_t to{(transformedRank + 1) * scaleDownFactor - 1};

      std::cout << "Add hit to interval [" << from << ", " << to << "] => " << result[transformedRank].second
                << " found" << std::endl;

#endif

      ++iii;
    }

    uint32_t currCount{0ULL};

#if LOG_DEBUG_RUN_TESTS
    std::cout << "=====================================" << std::endl;
    std::cout << "Final hit counts:" << std::endl;

    {
      size_t ii{0_z};
      for (auto&& r : result)
      {
        size_t from{ii * scaleDownFactor};
        size_t to{(ii + 1) * scaleDownFactor - 1};

        std::cout << "[" << from << ", " << to << "] => " << r.second << " found" << std::endl;

        ++ii;
      }
    }

    std::cout << "=====================================" << std::endl;
    std::cout << "Cumulative number of hits (as rank starting with 1):" << std::endl;
#endif

    {
      size_t ii{0_z};
      // Compute final chart values
      for (auto&& r : result)
      {
        uint32_t tmp{r.second};
        r.second = currCount;
        currCount += tmp;

#if LOG_DEBUG_RUN_TESTS
        size_t to{(ii)*scaleDownFactor};

        std::cout << "[0, " << to << "] => " << r.second << " images found" << std::endl;
#endif

        ++ii;
      }
    }

    vec_of_ranks.emplace_back(std::move(ranks));
    return result;
  }

  virtual ChartData RunModelTest(DataId data_ID, TransformationFunctionBase* pTransformFn,
                                 const std::vector<float>* pIndexKwFrequency,
                                 const std::vector<std::vector<UserImgQuery>>& testQueries,
                                 const std::vector<std::unique_ptr<Image>>& images,
                                 const std::map<eVocabularyId, KeywordsContainer>& keywordContainers) const override
  {
#if LOG_DEBUG_RUN_TESTS
    std::cout << "Running model test ... " << std::endl;
    std::cout << "Result chart will have " << MODEL_TEST_CHART_NUM_X_POINTS << " discrete points on X axis"
              << std::endl;
    std::cout << "=====================================" << std::endl;
#endif

    uint32_t maxRank = (uint32_t)images.size();

    // To have 100 samples
    uint32_t scaleDownFactor = maxRank / MODEL_TEST_CHART_NUM_X_POINTS;

    std::vector<std::pair<uint32_t, uint32_t>> result;
    result.resize(MODEL_TEST_CHART_NUM_X_POINTS + 1);

    uint32_t label{0ULL};
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
      for (auto&& [imgId, queryFormula, withExamples] : singleQuery)
      {
        formulae.push_back(queryFormula);
      }

      auto resultImages = GetRankedImages(formulae, data_ID, pTransformFn, pIndexKwFrequency, images,
                                          keywordContainers, 0ULL, imgId);

      // Rank index is -1 from rank
      size_t transformedRank = (resultImages.second - 1) / scaleDownFactor;
      // Increment this hit
      ++result[transformedRank].second;

#if LOG_DEBUG_RUN_TESTS
      std::cout << "----------------------------" << std::endl;
      std::cout << "Image ID " << imgId << "=> " << (resultImages.second - 1) << "/" << maxRank << std::endl;

      size_t from{transformedRank * scaleDownFactor};
      size_t to{(transformedRank + 1) * scaleDownFactor - 1};

      std::cout << "Add hit to interval [" << from << ", " << to << "] => " << result[transformedRank].second
                << " found" << std::endl;

#endif
    }

    uint32_t currCount{0ULL};

#if LOG_DEBUG_RUN_TESTS
    std::cout << "=====================================" << std::endl;
    std::cout << "Final hit counts:" << std::endl;

    {
      size_t ii{0_z};
      for (auto&& r : result)
      {
        size_t from{ii * scaleDownFactor};
        size_t to{(ii + 1) * scaleDownFactor - 1};

        std::cout << "[" << from << ", " << to << "] => " << r.second << " found" << std::endl;

        ++ii;
      }
    }

    std::cout << "=====================================" << std::endl;
    std::cout << "Cumulative number of hits (as rank starting with 1):" << std::endl;
#endif

    {
      size_t ii{0_z};
      // Compute final chart values
      for (auto&& r : result)
      {
        uint32_t tmp{r.second};
        r.second = currCount;
        currCount += tmp;

#if LOG_DEBUG_RUN_TESTS
        size_t to{(ii)*scaleDownFactor};

        std::cout << "[0, " << to << "] => " << r.second << " images found" << std::endl;
#endif

        ++ii;
      }
    }

    return result;
  }


  virtual std::pair<std::vector<size_t>, size_t> GetRankedImages(
      const std::vector<CnfFormula>& queryFormulae, DataId data_ID,
      TransformationFunctionBase* pTransformFn, const std::vector<float>* pIndexKwFrequency,
      const std::vector<std::unique_ptr<Image>>& images,
      const std::map<eVocabularyId, KeywordsContainer>& keywordContainers, size_t numResults,
      size_t targetImageId) const override
  {
    // If want all results
    if (numResults == SIZE_T_ERROR_VALUE)
    {
      numResults = images.size();
    }

    // Comparator lambda for priority queue
    auto cmp = [](const std::pair<float, size_t>& left, const std::pair<float, size_t>& right) {
      return left.first < right.first;
    };

    // Reserve enough space in container
    std::vector<std::pair<float, size_t>> container;
    container.reserve(images.size());

    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, decltype(cmp)> maxHeap(
        cmp, std::move(container));

    std::pair<std::vector<size_t>, size_t> result;
    result.first.reserve(numResults);

    // Check every image if satisfies query formula
    for (auto&& pImg : images)
    {
      //
      // Get ranking of main frame
      //

      // Get image ranking of main image
      auto imageRanking{GetImageQueryRanking(data_ID, pImg.get(), queryFormulae, pTransformFn, pIndexKwFrequency,
                                             images, keywordContainers)};

      if (imageRanking <= 0.0f)
      {
        // std::cout << "zero ranking!" << std::endl;
      }

      //
      // Count in temporal query if present
      //

      // If temporal query provided
      if (queryFormulae.size() > 1_z)
      {
        std::vector<CnfFormula> subFormulae;
        subFormulae.push_back(queryFormulae[1]);

        auto tempQueryRanking{GetImageTemporalQueryRanking(data_ID, pImg.get(), subFormulae, pTransformFn,
                                                           pIndexKwFrequency, images, keywordContainers)};

        if (tempQueryRanking <= 0.0f)
        {
          // std::cout << "zero ranking!" << std::endl;
        }

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
    size_t sizeHeap{maxHeap.size()};

    // Iterate over popping items in heap
    for (size_t i{0_z}; i < sizeHeap; ++i)
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

  
  float GetImageTemporalQueryRanking(DataId data_ID, Image* pImg,
                                     const std::vector<CnfFormula>& queryFormulae,
                                     TransformationFunctionBase* pTransformFn,
                                     const std::vector<float>* pIndexKwFrequency,
                                     const std::vector<std::unique_ptr<Image>>& images,
                                     const std::map<eVocabularyId, KeywordsContainer>& keywordContainers) const
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
    auto imgIt{images.begin() + pImg->m_index};

    // If item not found
    if (imgIt == images.end())
    {
      LOG_ERROR("Image not found in map.");
    }

    //
    // Iterate through all successors and compute their rankings
    //

    size_t count{0_z};
    // If this frame has some succesor frames
    for (size_t ii{0_z}; ii < pImg->m_numSuccessorFrames; ++ii)
    {
      if (count >= MAX_SUCC_CHECK_COUNT)
      {
        break;
      }

      auto bckpImgIt = imgIt;

      // Move iterator to successor
      ++imgIt;

      assert(imgIt != images.end());

      // Get image ranking of image
      auto imageRanking{GetImageQueryRanking(data_ID, imgIt->get(), queryFormulae, pTransformFn, pIndexKwFrequency,
                                             images, keywordContainers)};

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

#endif