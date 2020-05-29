#pragma once

#include <vector>

#include "common.h"
#include "utility.h"

namespace image_ranker
{
class BaseVectorTransform;
class KeywordsContainer;
class BaseImageset;

class [[nodiscard]] BaseSimUser
{
 public:
  // -----------------------------------------
  // We need virtual dctor.
  BaseSimUser() = default;
  BaseSimUser(const BaseSimUser& other) = default;
  BaseSimUser(BaseSimUser && other) = default;
  BaseSimUser& operator=(const BaseSimUser& other) = default;
  BaseSimUser& operator=(BaseSimUser&& other) = default;
  virtual ~BaseSimUser() noexcept = default;
  // -----------------------------------------

  virtual std::vector<UserTestQuery> process_sim_user(
      const BaseImageset* p_is, const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries, std::vector<ModelKeyValOption>& options) const = 0;

  virtual std::vector<UserTestQuery> generate_simulated_queries(
      [[maybe_unused]] const BaseImageset* p_is, [[maybe_unused]] const BaseVectorTransform& transformed_data,
      [[maybe_unused]] const KeywordsContainer& keywords, [[maybe_unused]] std::vector<ModelKeyValOption>& options,
      [[maybe_unused]] size_t count, [[maybe_unused]] size_t temporal_count = 1_z) const
  {
    std::string msg{ "Single queries generation not suported for this options combinations." };
    LOGW(msg);
    PROD_THROW_NOT_SUPP(msg);

    return std::vector<UserTestQuery>{};
  };
};

class [[nodiscard]] SimUserNoSim : public BaseSimUser
{
 public:
  [[nodiscard]] virtual std::vector<UserTestQuery> process_sim_user(
      const BaseImageset* p_is, const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries, std::vector<ModelKeyValOption>& options) const override;
};

class [[nodiscard]] SimUserXToP : public BaseSimUser
{
 public:
  /** Options for the given SimUser instance. */
  struct Options
  {
    float exponent_p{ DEF_SIM_USER_EXPONENT };
    eSimUserTarget target{ DEF_SIM_USER_TARGET };
    size_t num_words_from{ DEF_SIM_USER_NUM_WORDS_FROM };
    size_t num_words_to{ DEF_SIM_USER_NUM_WORDS_TO };
  };

  /**
   * Warning: Parsed pairs from \param options are removed.
   */
  [[nodiscard]] static Options parse_options(std::vector<ModelKeyValOption> & options)
  {
    std::vector<ModelKeyValOption> forwarded_options;

    Options opts;

    for (auto&& [key, val] : options)
    {
      if (key == "sim_user_num_words_from")
      {
        opts.num_words_from = strTo<size_t>(val);
      }
      else if (key == "sim_user_num_words_to")
      {
        opts.num_words_to = strTo<size_t>(val);
      }
      else if (key == "sim_user_paremeter_p")
      {
        opts.exponent_p = strTo<float>(val);
      }
      else if (key == "sim_user_target")
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
      // Forward those key-value pairs
      else
      {
        forwarded_options.emplace_back(key, val);
      }
    }

    options = forwarded_options;

    return opts;
  }

  [[nodiscard]] virtual std::vector<UserTestQuery> process_sim_user(
      const BaseImageset* p_is, const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries, std::vector<ModelKeyValOption>& options) const override;

  [[nodiscard]] virtual std::vector<UserTestQuery> generate_simulated_queries(
      const BaseImageset* p_is, const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      std::vector<ModelKeyValOption>& options, size_t count, size_t temporal_count = 1) const override;

 private:
  [[nodiscard]] std::vector<UserTestQuery> generate_whole_queries(
      const BaseImageset* p_is, const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries, const Options& options, size_t num_queries = 1) const;

  [[nodiscard]] std::vector<UserTestQuery> augment_with_temp_queries(
      const BaseImageset* p_is, const BaseVectorTransform& transformed_data, const KeywordsContainer& keywords,
      const std::vector<UserTestQuery>& test_user_queries, const Options& options, size_t count_additional_queries = 1)
      const;

  [[nodiscard]] CnfFormula generate_simulated_query(FrameId frame_ID, const BaseImageset* p_is,
                                                    const BaseVectorTransform& transformed_data,
                                                    const KeywordsContainer& keywords, const Options& options) const;
};

};  // namespace image_ranker