#pragma once

#include "BaseClassificationModel.h"

#include <common.h>
#include <queue>

extern std::vector<std::vector<size_t>> vec_of_ranks;


class ViretModel : public BaseClassificationModel
{
 public:
  /*!
   * List of possible settings what how to calculate rank
   */
  enum class eQueryOperations
  {
    cMultSum = 0,
    cMultMax = 1,
    cSumSum = 2,
    cSumMax = 3,
    cMaxMax = 4
  };

  struct Options
  {
    //! How to handle keyword frequency in ranking
    unsigned int m_keywordFrequencyHandling;

    //! What values are considered significant enough to calculate with them
    float m_trueTreshold;

    //! What operations are executed when creating rank for given image
    eQueryOperations m_queryOperation;

    eTempQueryOpOutter m_tempQueryOutterOperation;
    eTempQueryOpInner m_tempQueryInnerOperation;
  };

 public:
   static Options ParseOptionsString(const std::string& options_string);

    /**
   * Returns sorted vector of ranked images based on provided data for the given query.
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  virtual std::vector<FrameId> rank_frames(const Matrix<float>& data_mat, const KeywordsContainer& keywords,
                                           const std::vector<std::string>& user_query,
                                           const std::string& options = ""s) const override;

  /**
   * Returns results of this model after running provided test queries .
   *
   * Query in format: "1&3&4" where numbers are indices to scoring vector.
   */
  virtual std::vector<FrameId> run_test(
      const Matrix<float>& data_mat, const KeywordsContainer& keywords,
      const std::vector<std::pair<std::vector<std::string>, FrameId>>& test_user_queries,
      const std::string& options = ""s, size_t result_points = NUM_MODEL_TEST_RESULT_POINTS) const override;


};
