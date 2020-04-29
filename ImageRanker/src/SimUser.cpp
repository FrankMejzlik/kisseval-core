
#include "SimUser.h"

#include "datasets/BaseImageset.h"

using namespace image_ranker;

std::vector<UserTestQuery> SimUserNoSim::process_sim_user(const BaseImageset* p_is,
                                                          const BaseVectorTransform& transformed_data,
                                                          const KeywordsContainer& keywords,
                                                          const std::vector<UserTestQuery>& test_user_queries,
                                                          const std::vector<ModelKeyValOption>& options) const
{
  return test_user_queries;
}

std::vector<UserTestQuery> SimUserXToP::process_sim_user(const BaseImageset* p_is,
                                                         const BaseVectorTransform& transformed_data,
                                                         const KeywordsContainer& keywords,
                                                         const std::vector<UserTestQuery>& test_user_queries,
                                                         const std::vector<ModelKeyValOption>& options) const
{
  Options opts = parse_options(options);

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

std::vector<UserTestQuery> SimUserXToP::generate_whole_queries(const BaseImageset* p_is,
                                                                const BaseVectorTransform& transformed_data,
                                                                const KeywordsContainer& keywords,
                                                                const std::vector<UserTestQuery>& test_user_queries,
                                                                const Options& options, size_t num_queries) const
{
  LOGW("Not implemented!");
  throw NotSuportedModelOption("Single queries generation not suported for this options combinations.");
  return test_user_queries;
}

std::vector<UserTestQuery> SimUserXToP::augment_with_temp_queries(const BaseImageset* p_is,
                                                                  const BaseVectorTransform& transformed_data,
                                                                  const KeywordsContainer& keywords,
                                                                  const std::vector<UserTestQuery>& test_user_queries,
                                                                  const Options& options) const
{
  LOGW("Not implemented!");
  throw NotSuportedModelOption("Single queries generation not suported for this options combinations.");
  return test_user_queries;
}
