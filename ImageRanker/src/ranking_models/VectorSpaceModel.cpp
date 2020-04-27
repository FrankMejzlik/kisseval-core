
#include "VectorSpaceModel.h"

#include "BaseVectorTransform.h"
#include "utility.h"

using namespace image_ranker;

VectorSpaceModel::Options VectorSpaceModel::parse_options(const std::vector<ModelKeyValOption>& option_key_val_pairs)
{
  auto res{VectorSpaceModel::Options()};

  for (auto&& [key, val] : option_key_val_pairs)
  {
    // Model distance function
    if (key == enum_label(eModelOptsKeys::MODEL_DIST_FN).first)
    {
      if (val == "euclid")
      {
        res.dist_fn = eDistFunction::EUCLID_SQUARED;
      }
      else if (val == "cosine")
      {
        res.dist_fn = eDistFunction::COSINE;
      }
      else if (val == "manhattan")
      {
        res.dist_fn = eDistFunction::MANHATTAN;
      }
      else
      {
        LOGW("Unknown model option val '" + val + "'");
      }
    }
    else if (key == "model_IDF_method")
    {

    }
    else if (key == "model_IDF_method_true_threshold")
    {
      res.true_threshold = strTo<float>(val);
    }
    else if (key == "model_IDF_method_idf_coef")
    {
      res.idf_coef = strTo<float>(val);
    }
    // Weighing TERM term frequency
    else if (key == enum_label(eModelOptsKeys::MODEL_TERM_TF).first)
    {
      if (val == "natural")
      {
        res.term_tf = eTermFrequency::NATURAL;
      }
      else if (val == "log")
      {
        res.term_tf = eTermFrequency::LOGARIGHMIC;
      }
      else if (val == "augmented")
      {
        res.term_tf = eTermFrequency::AUGMENTED;
      }
      else
      {
        LOGW("Unknown model option val '" + val + "'");
      }
    }
    // Weighing TERM term frequency
    else if (key == enum_label(eModelOptsKeys::MODEL_TERM_IDF).first)
    {
      if (val == "none")
      {
        res.term_idf = eInvDocumentFrequency::NONE;
      }
      else if (val == "idf")
      {
        res.term_idf = eInvDocumentFrequency::IDF;
      }
      else
      {
        LOGW("Unknown model option val '" + val + "'");
      }
    }
    // Weighing TERM term frequency
    else if (key == enum_label(eModelOptsKeys::MODEL_QUERY_TF).first)
    {
      if (val == "natural")
      {
        res.query_tf = eTermFrequency::NATURAL;
      }
      else if (val == "log")
      {
        res.query_tf = eTermFrequency::LOGARIGHMIC;
      }
      else if (val == "augmented")
      {
        res.query_tf = eTermFrequency::AUGMENTED;
      }
      else
      {
        LOGW("Unknown model option val '" + val + "'");
      }
    }
    // Weighing TERM term frequency
    else if (key == enum_label(eModelOptsKeys::MODEL_QUERY_IDF).first)
    {
      if (val == "none")
      {
        res.query_idf = eInvDocumentFrequency::NONE;
      }
      else if (val == "idf")
      {
        res.query_idf = eInvDocumentFrequency::IDF;
      }
      else
      {
        LOGW("Unknown model option val '" + val + "'");
      }
    }
    else
    {
      LOGW("Unknown model option key '" + key + "'");
    }
  }

  return res;
}

RankingResult VectorSpaceModel::rank_frames(const BaseVectorTransform& transformed_data,
                                            const KeywordsContainer& keywords,
                                            const std::vector<CnfFormula>& user_query, size_t result_size,
                                            const std::vector<ModelKeyValOption>& options,
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

RankingResult VectorSpaceModel::rank_frames(const BaseVectorTransform& transformed_data,
                                            const KeywordsContainer& keywords,
                                            const std::vector<CnfFormula>& user_query, size_t result_size,
                                            const Options& opts, FrameId target_frame_ID) const
{
  using FramePair = std::pair<float, size_t>;

  // Comparator for the priority queue (we need min-heap)
  auto frame_pair_cmptor = [](const FramePair& left, const FramePair& right) { return left.first > right.first; };

  // Create inner container for the queue
  std::vector<FramePair> queue_cont;
  queue_cont.reserve(transformed_data.num_frames());

  // Construct prioprity queue
  std::priority_queue<FramePair, std::vector<FramePair>, decltype(frame_pair_cmptor)> max_prio_queue(
      frame_pair_cmptor, std::move(queue_cont));

  RankingResult result;
  result.target = target_frame_ID;
  result.m_frames.reserve(result_size);

  // Get cached data reference to already finally transformed data matrix
  const Matrix<float>& data_mat =
      transformed_data.data_sum_tfidf(opts.term_tf, opts.term_idf, opts.true_threshold, opts.idf_coef);
  // const Matrix<float>& data_mat = transformed_data.data_sum();

  auto dist_fn{get_dist_fn(opts.dist_fn)};

  // Create user query vector representation
  float query_t = opts.query_idf == eInvDocumentFrequency::IDF ? opts.true_threshold : 0.0F;
  const Vector<float>& idfs_vector{transformed_data.data_idfs(query_t, opts.idf_coef)};

  Vector<float> user_query_vec{
      create_user_query_vector(user_query.front(), transformed_data.num_dims(), idfs_vector, opts)};
  {
    // Iterate over all frame feature vectors
    size_t i{0};
    for (auto&& fea_vec : data_mat)
    {
      // Compute term weights
      const auto& frame_vector{fea_vec};

      // Ranking is distance in the space from the query
      float dist{dist_fn(user_query_vec, frame_vector)};

      max_prio_queue.emplace(dist, i);

      // \todo Add temporal ranking
      ++i;
    }
  }

  {
    bool found_target{target_frame_ID == ERR_VAL<FrameId>() ? true : false};
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
        result.m_frames.emplace_back();
      }
      max_prio_queue.pop();
    }
  }

  return result;
}

ModelTestResult VectorSpaceModel::test_model(const BaseVectorTransform& transformed_data,
                                             const KeywordsContainer& keywords,
                                             const std::vector<UserTestQuery>& test_user_queries,
                                             const std::vector<ModelKeyValOption>& options, size_t num_points) const
{
  size_t imageset_size{transformed_data.num_frames()};
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
  size_t i{0_z};
  for (auto&& [query, target_frame_ID] : test_user_queries)
  {
    //if (i % 100 == 0) LOG("i = " + std::to_string(i));

    auto res = rank_frames(transformed_data, keywords, query, 0, opts, target_frame_ID);

    uint32_t x{uint32_t(res.target_pos / divisor)};

    ++(test_results[x].second);
    ++i;
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

float VectorSpaceModel::rank_frame(const Vector<float>& frame_data, const CnfFormula& single_query,
                                   const Options& options) const
{
  float frame_ranking{1.0F};

  return frame_ranking;
}

Vector<float> VectorSpaceModel::create_user_query_vector(const CnfFormula& single_query, size_t vec_dim) const
{
  Vector<float> user_query_vec(vec_dim, 0.0F);

  for (auto&& clause : single_query)
  {
    float clause_ranking{0.0F};

    // \todo All subwords for now
    for (auto&& literal : clause)
    {
      // Set this idx to 1
      user_query_vec[literal.atom] = 1.0F;
    }
  }

  return user_query_vec;
}

Vector<float> VectorSpaceModel::create_user_query_vector(const CnfFormula& single_query, size_t vec_dim,
                                                         const Vector<float>& idfs, const Options& options) const
{
  Vector<float> user_query_vec(vec_dim, 0.0F);

  // Count term frequencies in the query
  Vector<size_t> term_frequencies(vec_dim, 0);
  size_t max_freq{0_z};
  for (auto&& clause : single_query)
  {
    for (auto&& literal : clause)
    {
      auto i{literal.atom};
      ++term_frequencies[i];

      max_freq = std::max(max_freq, term_frequencies[i]);
    }
  }

  /*
   * Choose TF scheme
   */
  auto tf_scheme_fn{pick_tf_scheme_fn(options.query_tf)};

  for (auto&& clause : single_query)
  {
    float clause_ranking{0.0F};

    for (auto&& literal : clause)
    {
      auto i{literal.atom};

      float tf = tf_scheme_fn(float(term_frequencies[i]), float(max_freq));
      float idf = idfs[i];

      if (options.query_idf == eInvDocumentFrequency::IDF)
      {
        user_query_vec[i] = tf * idf;
      }
      else
      {
        user_query_vec[i] = tf;
      }
    }
  }

  return user_query_vec;
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
