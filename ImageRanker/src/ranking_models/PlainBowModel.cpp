
#include "PlainBowModel.h"

#include "BaseVectorTransform.h"
#include "utility.h"

using namespace image_ranker;

PlainBowModel::Options PlainBowModel::parse_options(const std::vector<ModelKeyValOption>& option_key_val_pairs)
{
  auto res{PlainBowModel::Options()};

  for (auto&& [key, val] : option_key_val_pairs)
  {
    if (key == enum_label(eModelOptsKeys::MODEL_SUB_PCA_MEAN).first)
    {
      if (val == "true")
      {
        res.sub_PCA_mean = true;
      }
    }
    if (key == enum_label(eModelOptsKeys::MODEL_DIST_FN).first)
    {
      if (val == "cosine")
      {
        res.dist_fn = eDistFunction::COSINE_NONORM;
      }
      else if (val == "manhattan")
      {
        res.dist_fn = eDistFunction::MANHATTAN;
      }
      else if (val == "euclid")
      {
        res.dist_fn = eDistFunction::EUCLID;
      }
      else
      {
        LOGW("Unknown model option value '" + val + "'");
      }
    }
    else
    {
      LOGW("Unknown model option key '" + key + "'");
    }
  }

  return res;
}

RankingResult PlainBowModel::rank_frames(const Matrix<float>& transformed_data, const Matrix<float>& kw_features,
                                         const Vector<float>& kw_bias_vec, const Matrix<float>& kw_PCA_mat,
                                         const Vector<float>& kw_PCA_mean_vec, const KeywordsContainer& keywords,
                                         const std::vector<CnfFormula>& user_query, size_t result_size,
                                         const std::vector<ModelKeyValOption>& options, FrameId target_frame_ID) const
{
  if (user_query.empty())
  {
    LOGW("Empty query");
    return RankingResult{};
  }

  // Parse provided options
  Options opts = parse_options(options);

  return rank_frames(transformed_data, kw_features, kw_bias_vec, kw_PCA_mat, kw_PCA_mean_vec, keywords, user_query,
                     result_size, opts, target_frame_ID);
}

Vector<float> PlainBowModel::embedd_native_user_query(const Matrix<float>& kw_features,
                                                      const Vector<float>& kw_bias_vec, const Matrix<float>& kw_PCA_mat,
                                                      const Vector<float>& kw_PCA_mean_vec, const CnfFormula& query,
                                                      const Options& opts) const
{
  // Initialize zero vector
  std::vector<float> score_vec(kw_features.front().size(), 0.0F);

  // Accumuate scores for given keywords
  for (auto&& clause : query)
  {
    auto kw_ID{clause.front().atom};
    score_vec = vec_add(score_vec, kw_features[kw_ID]);
  }

  // Add bias
  score_vec = vec_add(score_vec, kw_bias_vec);

  // Apply hyperbolic tangent function
  std::transform(score_vec.begin(), score_vec.end(), score_vec.begin(), [](const float& score) { return tanh(score); });

  if (opts.do_PCA)
  {
    // Normalize
    score_vec = normalize(score_vec);

    if (opts.sub_PCA_mean)
    {
      // Sub mean vec
      score_vec = vec_sub(score_vec, kw_PCA_mean_vec);
    }

    // Do PCA
    score_vec = mat_vec_prod(kw_PCA_mat, score_vec);
  }

  // Normalize
  score_vec = normalize(score_vec);

  return score_vec;
}

RankingResult PlainBowModel::rank_frames(const Matrix<float>& data_mat, const Matrix<float>& kw_features,

                                         const Vector<float>& kw_bias_vec, const Matrix<float>& kw_PCA_mat,
                                         const Vector<float>& kw_PCA_mean_vec, 
  [[maybe_unused]] const KeywordsContainer& keywords,
                                         const std::vector<CnfFormula>& user_query, size_t result_size,
                                         const Options& opts, FrameId target_frame_ID) const
{
  using FramePair = std::pair<float, size_t>;

  size_t num_frames{ data_mat.size() };

  // Check valid target ID
  if (target_frame_ID >= num_frames)
  {
    std::string msg{"Invalid `target_frame_ID` = " + std::to_string(target_frame_ID) + "."};
    LOGE(msg);
    PROD_THROW("Invalid parameters in function call.");
  }

  // Adjust desired result size
  result_size = std::min(result_size, num_frames);

  // Comparator for the priority queue
  auto frame_pair_cmptor = [](const FramePair& left, const FramePair& right) { return left.first > right.first; };

  // Create inner container for the queue
  std::vector<FramePair> queue_cont;
  queue_cont.reserve(data_mat.size());

  // Construct prioprity queue
  std::priority_queue<FramePair, std::vector<FramePair>, decltype(frame_pair_cmptor)> max_prio_queue(
      frame_pair_cmptor, std::move(queue_cont));

  RankingResult result;
  result.target = target_frame_ID;
  result.m_frames.reserve(result_size);

  std::vector<Vector<float>> embedded_user_queries;

  size_t visual_dim{data_mat.front().size()};
  size_t text_dim{kw_features.front().size()};

  // Copy options
  Options new_opts {opts};

  // If only PCAed visual data available, force PCA
  if (visual_dim < text_dim)
  {
    new_opts.do_PCA = true;
  }

  for (auto&& q : user_query)
  {
    auto emb_q{embedd_native_user_query(kw_features, kw_bias_vec, kw_PCA_mat, kw_PCA_mean_vec, q, new_opts)};
    embedded_user_queries.emplace_back(emb_q);

    assert(emb_q.size() == visual_dim);
  }


  auto dist_fn{get_dist_fn(new_opts.dist_fn)};

  {
    size_t i{0};
    for (auto&& fea_vec : data_mat)
    {
      // Ranking is distance in the space from the query
      float dist{dist_fn(embedded_user_queries.front(), fea_vec)};

      max_prio_queue.emplace(dist, i);

      // \todo Add temporal ranking
      if (user_query.size() > 1)
      {
        throw NotSuportedModelOptionExcept("Temporal queries not yet suported with this model.");
      }
      ++i;
    }
  }

  {
    bool found_target{target_frame_ID == ERR_VAL<FrameId>()};
    for (size_t i{0}; i < result_size || !found_target; ++i)
    {
      auto pair{max_prio_queue.top()};

      if (pair.second == target_frame_ID)
      {
        found_target = true;
        result.target_pos = i + 1;
      }

      if (i < result_size)
      {
        result.m_frames.emplace_back(FrameId(pair.second));
      }
      max_prio_queue.pop();
    }
  }

  return result;
}

ModelTestResult PlainBowModel::test_model(const Matrix<float>& transformed_data, const Matrix<float>& kw_features,
                                          const Vector<float>& kw_bias_vec, const Matrix<float>& kw_PCA_mat,
                                          const Vector<float>& kw_PCA_mean_vec, const KeywordsContainer& keywords,
                                          const std::vector<UserTestQuery>& test_user_queries,
                                          const std::vector<ModelKeyValOption>& options, size_t num_points) const
{
  size_t imageset_size{transformed_data.size()};
  float divisor{float(imageset_size) / num_points};

  // Prefil result [x,f(x)] values
  std::vector<std::pair<uint32_t, uint32_t>> test_results;
  test_results.reserve(num_points + 1);
  for (size_t i{0}; i <= num_points; ++i)
  {
    test_results.emplace_back(uint32_t(i * divisor), 0);
  }

  // Parse provided options
  Options opts = parse_options(options);

  // Rank them all
  for (auto&& [query, target_frame_ID] : test_user_queries)
  {
    auto res = rank_frames(transformed_data, kw_features, kw_bias_vec, kw_PCA_mat, kw_PCA_mean_vec, keywords, query, 0,
                           opts, target_frame_ID);

    uint32_t x{uint32_t(res.target_pos / divisor)};

    ++(test_results[x].second);
  }

  // Sumarize results into results
  uint32_t sum{0};
  for (auto&& num_hits : test_results)
  {
    auto val{num_hits.second};
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
