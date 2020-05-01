
#include "MultSumMaxModel.h"

#include "BaseVectorTransform.h"
#include "utility.h"

using namespace image_ranker;

MultSumMaxModel::Options MultSumMaxModel::parse_options(const std::vector<ModelKeyValOption>& option_key_val_pairs)
{
  auto res{ MultSumMaxModel::Options() };

  for (auto&& [key, val] : option_key_val_pairs)
  {
    if (key == enum_label(eModelOptsKeys::MODEL_OPERATIONS).first)
    {
      if (val == "mult-sum")
      {
        res.scoring_operations = eScoringOperations::cMultSum;
      }
      else if (val == "mult-max")
      {
        res.scoring_operations = eScoringOperations::cMultMax;
      }
      else if (val == "sum-sum")
      {
        res.scoring_operations = eScoringOperations::cSumSum;
      }
      else if (val == "sum-max")
      {
        res.scoring_operations = eScoringOperations::cSumMax;
      }
      else if (val == "max-max")
      {
        res.scoring_operations = eScoringOperations::cMaxMax;
      }
      else
      {
        LOGW("Unknown model option value for key '" + key + "' -> '" + val + "'");
      }
    }
    else if (key == enum_label(eModelOptsKeys::MODEL_INNER_OP).first)
    {
      if (val == "sum")
      {
        res.succ_aggregation = eSuccesorAggregation::cSum;
      }
      else if (val == "max")
      {
        res.succ_aggregation = eSuccesorAggregation::cMax;
      }
      else if (val == "mult")
      {
        res.succ_aggregation = eSuccesorAggregation::cProduct;
      }
      else
      {
        LOGW("Unknown model option value for key '" + key + "' -> '" + val + "'");
      }
    }
    else if (key == enum_label(eModelOptsKeys::MODEL_OUTTER_OP).first)
    {
      if (val == "sum")
      {
        res.main_temp_aggregation = eMainTempRankingAggregation::cSum;
      }
      else if (val == "mult")
      {
        res.main_temp_aggregation = eMainTempRankingAggregation::cProduct;
      }
      else
      {
        LOGW("Unknown model option value for key '" + key + "' -> '" + val + "'");
      }
    }
    else if (key == enum_label(eModelOptsKeys::MODEL_IGNORE_THRESHOLD).first)
    {
      res.ignore_below_threshold = strTo<float>(val);
    }
    else
    {
      LOGW("Unknown model option key '" + key + "'");
    }
  }

  return res;
}

RankingResult MultSumMaxModel::rank_frames(const BaseVectorTransform& transformed_data,
                                           const KeywordsContainer& keywords, const std::vector<CnfFormula>& user_query,
                                           size_t result_size, const std::vector<ModelKeyValOption>& options,
                                           FrameId target_frame_ID) const
{
  if (user_query.empty())
  {
    LOGW("Empty query");
    return RankingResult{};
  }

  // Parse provided options
  Options opts = parse_options(options);

  return rank_frames(transformed_data, keywords, user_query, result_size, opts, target_frame_ID);
}

RankingResult MultSumMaxModel::rank_frames(const BaseVectorTransform& transformed_data,
                                           const KeywordsContainer& keywords, const std::vector<CnfFormula>& user_query,
                                           size_t result_size, const Options& opts, FrameId target_frame_ID) const
{
  using FramePair = std::pair<float, FrameId>;

  // Comparator for the priority queue
  auto frame_pair_cmptor = [](const FramePair& left, const FramePair& right) { return left.first < right.first; };

  // Create inner container for the queue
  std::vector<FramePair> queue_cont;
  queue_cont.reserve(transformed_data.num_frames());

  // Construct prioprity queue
  std::priority_queue<FramePair, std::vector<FramePair>, decltype(frame_pair_cmptor)> max_prio_queue(
      frame_pair_cmptor, std::move(queue_cont));

  RankingResult result;
  result.target = target_frame_ID;
  result.m_frames.reserve(result_size);

  const Matrix<float>* p_data_mat;
  switch (opts.scoring_operations)
  {
      // SUM based
    case eScoringOperations::cMultSum:
    case eScoringOperations::cSumSum:
      p_data_mat = &transformed_data.data_sum();
      break;

      // MAX based
    case eScoringOperations::cMaxMax:
    case eScoringOperations::cMultMax:
    case eScoringOperations::cSumMax:
      p_data_mat = &transformed_data.data_max();
      break;

    default:
      LOGE("Unknown scoring operation.");
      return RankingResult{};
  }

  const Matrix<float>& data_mat{ *p_data_mat };

  {
    FrameId i{ 0 };
    for (auto&& fea_vec : data_mat)
    {
      float prim_ranking = rank_frame(fea_vec, user_query.front(), opts);

      // \todo Add temporal dynamic temp ranking
      if (user_query.size() > 1)
      {
        auto q2{ user_query[1] };

        float secondary_ranking{};

        switch (opts.succ_aggregation)
        {
          case eSuccesorAggregation::cMax:
          case eSuccesorAggregation::cSum:
            secondary_ranking = 0.0F;
            break;

          case eSuccesorAggregation::cProduct:
            secondary_ranking = 1.0F;

          default:
            LOGW("Uknown option!");
            break;
        }

        for (size_t ii{ 0_z }; ii < TEMP_CONTEXT_LOOKUP_LENGTH && (ii + i) < data_mat.size(); ++ii)
        {
          auto features_vec{ data_mat[ii + i] };

          float ranking = rank_frame(features_vec, q2, opts);

          switch (opts.succ_aggregation)
          {
            case eSuccesorAggregation::cMax:
              secondary_ranking = std::max(secondary_ranking, ranking);
              break;

            case eSuccesorAggregation::cSum:
              secondary_ranking += ranking;
              break;

            case eSuccesorAggregation::cProduct:
              secondary_ranking *= ranking;

            default:
              LOGW("Uknown option!");
              break;
          }
        }

        // Combine with primary ranking
        if (opts.main_temp_aggregation == eMainTempRankingAggregation::cProduct)
        {
          prim_ranking = prim_ranking * secondary_ranking;
        }
        else if (opts.main_temp_aggregation == eMainTempRankingAggregation::cSum)
        {
          prim_ranking = prim_ranking + secondary_ranking;
        }
      }

      max_prio_queue.emplace(prim_ranking, i);

      ++i;
    }
  }

  {
    bool found_target{ target_frame_ID == ERR_VAL<FrameId>() ? true : false };
    for (size_t i{ 0 }; i < result_size || !found_target; ++i)
    {
      auto pair{ max_prio_queue.top() };

      if (pair.second == target_frame_ID)
      {
        found_target = true;
        result.target_pos = i + 1;
      }

      if (i < result_size)
      {
        result.m_frames.emplace_back(pair.second);
      }
      max_prio_queue.pop();
    }
  }

  return result;
}

ModelTestResult MultSumMaxModel::test_model(const BaseVectorTransform& transformed_data,
                                            const KeywordsContainer& keywords,
                                            const std::vector<UserTestQuery>& test_user_queries,
                                            const std::vector<ModelKeyValOption>& options, size_t num_points) const
{
  size_t imageset_size{ transformed_data.num_frames() };
  float divisor{ float(imageset_size) / num_points };

  // Prefil result [x,f(x)] values
  std::vector<std::pair<uint32_t, uint32_t>> test_results;
  test_results.reserve(num_points + 1);
  for (size_t i{ 0 }; i <= num_points; ++i)
  {
    test_results.emplace_back(uint32_t(i * divisor), 0);
  }

  // Parse provided options
  Options opts = parse_options(options);

  // Rank them all
  for (auto&& [query, target_frame_ID] : test_user_queries)
  {
    auto res = rank_frames(transformed_data, keywords, query, 0, opts, target_frame_ID);

    uint32_t x{ uint32_t(res.target_pos / divisor) };

    ++(test_results[x].second);
  }

  // Sumarize results into results
  uint32_t sum{ 0 };
  for (auto&& num_hits : test_results)
  {
    auto val{ num_hits.second };
    num_hits.second = sum;
    sum += val;
  }

  return test_results;
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
    if (numResults == ERR_VAL<size_t>())
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

float MultSumMaxModel::rank_frame(const Vector<float>& frame_data, const CnfFormula& single_query,
                                  const Options& options) const
{
  float frame_ranking{ 1.0F };

  /********************************************************
   * SETTINGS: Initialize frame ranking with correct value
   ********************************************************/
  switch (options.scoring_operations)
  {
    case eScoringOperations::cMultMax:
    case eScoringOperations::cMultSum:
      frame_ranking = 1.0F;
      break;

    case eScoringOperations::cSumSum:
    case eScoringOperations::cSumMax:
    case eScoringOperations::cMaxMax:
      frame_ranking = 0.0F;
      break;

    default:
      LOGE("Unknown scoring operation optaion: '"s + std::to_string(int(options.scoring_operations)) + "'."s);
  }

  for (auto&& clause : single_query)
  {
    float clause_ranking{ 0.0F };

    for (auto&& literal : clause)
    {
      float literal_ranking{ frame_data[literal.atom] };

      // If literal_ranking under the threshold
      if (literal_ranking < options.ignore_below_threshold)
      {
        continue;
      }

      /********************************************************
       * SETTINGS: Choose correct INNER query operation
       ********************************************************/
      switch (options.scoring_operations)
      {
          // Inner operation: SUM
        case eScoringOperations::cMultSum:
        case eScoringOperations::cSumSum:
          clause_ranking += literal_ranking;
          break;

          // Inner operation: MAX
        case eScoringOperations::cMultMax:
        case eScoringOperations::cSumMax:
        case eScoringOperations::cMaxMax:
          clause_ranking = std::max(clause_ranking, literal_ranking);
          break;

        default:
          LOGE("Unknown query operation "s + std::to_string(int(options.scoring_operations)) + "."s);
      }
    }

    /********************************************************
     * SETTINGS: Choose correct OUTTER query operation
     ********************************************************/
    switch (options.scoring_operations)
    {
        // Outter operationn: MULT
      case eScoringOperations::cMultSum:
      case eScoringOperations::cMultMax:
      {
        frame_ranking *= clause_ranking;
      }
      break;

        // Outter operationn: SUM
      case eScoringOperations::cSumSum:
      case eScoringOperations::cSumMax:
      {
        frame_ranking += clause_ranking;
      }
      break;

        // Outter operation: MAX
      case eScoringOperations::cMaxMax:
      {
        frame_ranking = std::max(frame_ranking, clause_ranking);
      }
      break;

      default:
        LOGE("Unknown query scoring operation: '"s + std::to_string(int(options.scoring_operations)) + "'."s);
    }
  }

  return frame_ranking;
}