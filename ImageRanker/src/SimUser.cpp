
#include "SimUser.h"

#include "BaseVectorTransform.h"
#include "datasets/BaseImageset.h"

using namespace image_ranker;

std::vector<UserTestQuery> SimUserNoSim::process_sim_user(
    [[maybe_unused]] const BaseImageset* p_is, [[maybe_unused]] const BaseVectorTransform& transformed_data,
    [[maybe_unused]] const KeywordsContainer& keywords, const std::vector<UserTestQuery>& test_user_queries,
    [[maybe_unused]] std::vector<ModelKeyValOption>& options) const
{
  return test_user_queries;
}

std::vector<UserTestQuery> SimUserXToP::process_sim_user(const BaseImageset* p_is,
                                                         const BaseVectorTransform& transformed_data,
                                                         const KeywordsContainer& keywords,
                                                         const std::vector<UserTestQuery>& test_user_queries,
                                                         std::vector<ModelKeyValOption>& ref_options) const
{
  Options opts = parse_options(ref_options);

  switch (opts.target)
  {
    case eSimUserTarget::SINGLE_QUERIES:
      return generate_whole_queries(p_is, transformed_data, keywords, test_user_queries, opts);
      break;

    case eSimUserTarget::TEMP_QUERIES:
      return generate_whole_queries(p_is, transformed_data, keywords, test_user_queries, opts, 2_z);
      break;

    case eSimUserTarget::AUGMENT_REAL_WITH_TEMP:
      return augment_with_temp_queries(p_is, transformed_data, keywords, test_user_queries, opts);
      break;

    default:
      LOGW("Unknown sim user target: " << int(opts.target));
      break;
  }

  return test_user_queries;
}

std::vector<UserTestQuery> SimUserXToP::generate_simulated_queries(const BaseImageset* p_is,
                                                                   const BaseVectorTransform& transformed_data,
                                                                   const KeywordsContainer& keywords,
                                                                   std::vector<ModelKeyValOption>& options,
                                                                   size_t count, size_t temporal_count) const
{
  Options opts = parse_options(options);

  std::random_device rd;
  std::mt19937 generator{ rd() };
  std::uniform_int_distribution<FrameId> disr{ FrameId(0), FrameId(transformed_data.num_frames() - 1) };

  std::vector<UserTestQuery> result_queries{};
  result_queries.reserve(count);

  for (size_t i{ 0_z }; i < count; ++i)
  {
    FrameId fr_ID = disr(generator);

    std::vector<CnfFormula> test_CNF_formulas;
    test_CNF_formulas.reserve(temporal_count + 1);

    for (size_t j{ 0_z }; j < temporal_count; ++j)
    {
      test_CNF_formulas.emplace_back(
          generate_simulated_query(FrameId(fr_ID + j), p_is, transformed_data, keywords, opts));
    }

    result_queries.emplace_back(test_CNF_formulas, fr_ID);
  }

  return result_queries;
}

std::vector<UserTestQuery> SimUserXToP::generate_whole_queries(const BaseImageset* p_is,
                                                               const BaseVectorTransform& transformed_data,
                                                               const KeywordsContainer& keywords,
                                                               const std::vector<UserTestQuery>& test_user_queries,
                                                               const Options& options, size_t temporal_count) const
{
  std::random_device rd;
  std::mt19937 generator{ rd() };
  std::uniform_int_distribution<FrameId> disr{ FrameId(0), FrameId(transformed_data.num_frames() - 1) };

  std::vector<UserTestQuery> result_queries{};

  for (size_t i{ 0_z }; i < test_user_queries.size(); ++i)
  {
    auto real_user_q = test_user_queries[i];
    FrameId fr_ID = real_user_q.second;

    std::vector<CnfFormula> test_CNF_formulas;
    test_CNF_formulas.reserve(temporal_count + 1);

    for (size_t j{ 0_z }; j < temporal_count; ++j)
    {
      test_CNF_formulas.emplace_back(
          generate_simulated_query(FrameId(fr_ID + j), p_is, transformed_data, keywords, options));
    }

    result_queries.emplace_back(test_CNF_formulas, fr_ID);
  }

  return result_queries;
}

std::vector<UserTestQuery> SimUserXToP::augment_with_temp_queries(
    const BaseImageset* p_is, const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
    const std::vector<UserTestQuery>& test_user_queries, const Options& options, size_t count_additional_queries) const
{
  std::random_device rd;
  std::mt19937 generator{ rd() };
  std::uniform_int_distribution<FrameId> disr{ FrameId(0), FrameId(transformed_data.num_frames() - 1) };

  std::vector<UserTestQuery> result_queries{};

  for (size_t i{ 0_z }; i < test_user_queries.size(); ++i)
  {
    auto real_user_q = test_user_queries[i];
    FrameId fr_ID = real_user_q.second;
    std::vector<CnfFormula> query = real_user_q.first;

    for (size_t j{ 0_z }; j < count_additional_queries; ++j)
    {
      query.emplace_back(generate_simulated_query(FrameId(fr_ID + j), p_is, transformed_data, keywords, options));
    }

    result_queries.emplace_back(query, fr_ID);
  }

  return result_queries;
}

CnfFormula SimUserXToP::generate_simulated_query(FrameId frame_ID, const BaseImageset* p_is,
                                                 const BaseVectorTransform& transformed_data,
                                                 const KeywordsContainer& keywords, const Options& options) const
{
  auto frame_fea_vec{ transformed_data.data_linear_raw()[frame_ID] };

  size_t from{ options.num_words_from };
  size_t to{ options.num_words_to };
  auto epxonent{ options.exponent_p };

  UserTestQuery res_query;

  auto pImgData{ p_is->operator[](frame_ID) };

  const auto& linBinVector{ frame_fea_vec };

  // Calculate transformed vector
  float totalSum{ 0.0F };
  std::vector<float> transformedData;
  for (auto&& value : linBinVector)
  {
    float newValue{ pow(value, epxonent) };
    transformedData.push_back(newValue);

    totalSum += newValue;
  }

  // Get scaling coef
  float scaleCoef{ 1 / totalSum };

  // Normalize
  size_t i{ 0_z };
  float cummulSum{ 0.0F };
  for (auto&& value : transformedData)
  {
    cummulSum += value * scaleCoef;

    transformedData[i] = cummulSum;

    ++i;
  }

  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  auto randLabel{ distribution(generator) };

  size_t numLabels{ size_t((randLabel * (to - from)) + from) };

  std::vector<size_t> queryLabels;
  for (size_t i{ 0_z }; i < numLabels; ++i)
  {
    // Get random number between [0, 1] from uniform distribution
    float rand{ distribution(generator) };
    size_t labelIndex{ 0_z };

    // Iterate through discrete points while we haven't found correct point
    while (transformedData[labelIndex] < rand)
    {
      ++labelIndex;
    }

    queryLabels.push_back(labelIndex);
  }

  std::vector<Clause> queryFormula;
  queryFormula.reserve(numLabels);

  // Create final formula with target indices
  for (auto&& index : queryLabels)
  {
    queryFormula.emplace_back(Clause{ Literal<KeywordId>{ index, false } });
  }

  return queryFormula;
}
