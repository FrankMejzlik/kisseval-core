#pragma once

#include <vector>

#include "common.h"
#include "utility.h"

namespace image_ranker
{
class BaseVectorTransform;
class KeywordsContainer;
class BaseImageset;

enum class eSimUserTarget
{
  SINGLE_QUERIES,
  TEMP_QUERIES,
  AUGMENT_REAL_WITH_TEMP,
  _COUNT
};

class BaseSimUser
{
 public:
  virtual std::vector<UserTestQuery> process_sim_user(const BaseImageset* p_is,
                                                      const BaseVectorTransform& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<UserTestQuery>& test_user_queries,
                                                      const std::vector<ModelKeyValOption>& options) const = 0;
};

class SimUserNoSim : public BaseSimUser
{
 public:
  virtual std::vector<UserTestQuery> process_sim_user(const BaseImageset* p_is,
                                                      const BaseVectorTransform& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<UserTestQuery>& test_user_queries,
                                                      const std::vector<ModelKeyValOption>& options) const override;
};

class SimUserXToP : public BaseSimUser
{
 public:
  struct Options
  {
    Options() : exponent_p(5.0F), target(eSimUserTarget::SINGLE_QUERIES) {}

    float exponent_p;
    eSimUserTarget target;
  };

  static Options parse_options(const std::vector<ModelKeyValOption>& options)
  {
    Options opts;

    for (auto&& [key, val] : options)
    {
      if (key == "paremeter_p")
      {
        opts.exponent_p = strTo<float>(val);
      }
      else if (key == "target")
      {
        if (val == "single_queries")
        {
          opts.target = eSimUserTarget::SINGLE_QUERIES;
        }
        else if (val == "temp_queries")
        {
          opts.target = eSimUserTarget::TEMP_QUERIES;
        }
        else if (val == "alter_real_with_temporal")
        {
          opts.target = eSimUserTarget::AUGMENT_REAL_WITH_TEMP;
        }
        else
        {
          LOGW("Uknown value '" + val + "'.")
        }
      }
    }

    return opts;
  }

  virtual std::vector<UserTestQuery> process_sim_user(const BaseImageset* p_is,
                                                      const BaseVectorTransform& transformed_data,
                                                      const KeywordsContainer& keywords,
                                                      const std::vector<UserTestQuery>& test_user_queries,
                                                      const std::vector<ModelKeyValOption>& options) const override;

 private:
  std::vector<UserTestQuery> generate_single_queries(const BaseImageset* p_is,
                                                     const BaseVectorTransform& transformed_data,
                                                     const KeywordsContainer& keywords,
                                                     const std::vector<UserTestQuery>& test_user_queries,
                                                     const Options& options, size_t num_queries = 1_z) const;

  std::vector<UserTestQuery> augment_with_temp_queries(const BaseImageset* p_is,
                                                       const BaseVectorTransform& transformed_data,
                                                       const KeywordsContainer& keywords,
                                                       const std::vector<UserTestQuery>& test_user_queries,
                                                       const Options& options) const;
};

};  // namespace image_ranker