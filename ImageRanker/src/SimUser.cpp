
#include "SimUser.h"

using namespace image_ranker;

std::vector<UserTestQuery> SimUserNoSim::process_sim_user(const BaseVectorTransform& transformed_data,
                                                          const KeywordsContainer& keywords,
                                                          const std::vector<UserTestQuery>& test_user_queries,
                                                          const std::vector<ModelKeyValOption>& options) const
{
  return test_user_queries;
}

std::vector<UserTestQuery> SimUserSingleQueries::process_sim_user(const BaseVectorTransform& transformed_data,
                                                                  const KeywordsContainer& keywords,
                                                                  const std::vector<UserTestQuery>& test_user_queries,
                                                                  const std::vector<ModelKeyValOption>& options) const
{
  // \todo Implement.
  return test_user_queries;
}

std::vector<UserTestQuery> SimUserTempQueries::process_sim_user(const BaseVectorTransform& transformed_data,
                                                                const KeywordsContainer& keywords,
                                                                const std::vector<UserTestQuery>& test_user_queries,
                                                                const std::vector<ModelKeyValOption>& options) const
{
  // \todo Implement.
  return test_user_queries;
}

std::vector<UserTestQuery> SimUserAugmentRealWithTemp::process_sim_user(
    const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
    const std::vector<UserTestQuery>& test_user_queries, const std::vector<ModelKeyValOption>& options) const
{
  // \todo Implement.
  return test_user_queries;
}