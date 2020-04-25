#pragma once

#include <vector>

#include "common.h"

namespace image_ranker
{
class BaseVectorTransform;
class KeywordsContainer;

class BaseSimUser
{
 public:
  virtual std::vector<UserTestQuery> process_sim_user(const BaseVectorTransform& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<UserTestQuery>& test_user_queries,
                                                      const std::vector<ModelKeyValOption>& options) const = 0;
};

class SimUserNoSim : public BaseSimUser
{
 public:
  virtual std::vector<UserTestQuery> process_sim_user(const BaseVectorTransform& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<UserTestQuery>& test_user_queries,
                                                      const std::vector<ModelKeyValOption>& options) const override;
};

class SimUserSingleQueries : public BaseSimUser
{
 public:
  virtual std::vector<UserTestQuery> process_sim_user(const BaseVectorTransform& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<UserTestQuery>& test_user_queries,
                                                      const std::vector<ModelKeyValOption>& options) const override;
};

class SimUserTempQueries : public BaseSimUser
{
  virtual std::vector<UserTestQuery> process_sim_user(const BaseVectorTransform& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<UserTestQuery>& test_user_queries,
                                                      const std::vector<ModelKeyValOption>& options) const override;
};

class SimUserAugmentRealWithTemp : public BaseSimUser
{
 public:
  virtual std::vector<UserTestQuery> process_sim_user(const BaseVectorTransform& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<UserTestQuery>& test_user_queries,
                                                      const std::vector<ModelKeyValOption>& options) const override;
};

};  // namespace image_ranker