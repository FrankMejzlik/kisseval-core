#pragma once

#include "common.h"

#include "Database.h"

namespace image_ranker
{
class ImageRanker;

struct SearchSessionAction
{
  size_t session_ID;
  size_t action_index;
  size_t rank;
};

struct SearchSession
{
  size_t ID;
  FrameId target_frame_ID;
  size_t duration;
  bool found;
  std::vector<SearchSessionAction> actions;
};

class DataManager
{
 public:
  DataManager(ImageRanker* p_owner);

  void submit_annotator_user_queries(const StringId& data_pack_ID, const StringId& vocab_ID,
                                     const ::std::string& model_options, size_t user_level, bool with_example_images,
                                     const std::vector<AnnotatorUserQuery>& user_queries);

  [[nodiscard]] std::vector<UserTestQuery> fetch_user_test_queries(eUserQueryOrigin queries_origin,
                                                                   const StringId& vocabulary_ID,
                                                                   const StringId& data_pack_ID = ""s,
                                                                   const StringId& model_options = ""s) const;

  [[nodiscard]] std::vector<UserTestNativeQuery> fetch_user_native_test_queries(eUserQueryOrigin queries_origin) const;

  bool submit_search_session(const std::string& data_pack_ID, const std::string& vocabulary_ID,
                             const std::string& model_commands, size_t user_level, bool with_example_images,
                             FrameId target_frame_ID, eSearchSessionEndStatus end_status, size_t duration,
                             const std::string& sessionId, const std::vector<InteractiveSearchAction>& actions);

  /**
   * Returns struct containing chart data showing Q1, Q2, Q3 for progress of ranks for currently collected search
   * sessions.
   *
   * \param   data_pack_ID    Collected data over this `data_pack_ID` will be used for the chart generation.
   * \param   model_options   Additional specification on what data will be used as source data.
   * \param   max_user_level  Maximum user level records that will be used for generating the data.
   * \return                  Struct with data needed for plotting the chart.
   */
  [[nodiscard]] SearchSessRankChartData get_search_sessions_rank_progress_chart_data(
      const std::string& data_pack_ID, const std::string& model_options, size_t max_user_level, size_t num_frames,
      size_t min_samples_count, bool normalize) const;

 private:
  [[nodiscard]] QuantileLineChartData<size_t, float> get_aggregate_rank_progress_data(
      const std::vector<SearchSession>& sessions, size_t max_sess_len, size_t num_frames_total,
      size_t min_samples_count, bool normalize) const;
  [[nodiscard]] MedianLineMultichartData<size_t, float> get_median_multichart_rank_progress_data(
      const std::vector<SearchSession>& sessions, size_t max_sess_len, size_t num_frames_total,
      size_t min_samples_count, bool normalize) const;

 private:
  ImageRanker* _p_owner;
  Database _db;

  const std::string queries_table_name = "user_queries";
  const std::string searches_table_name = "search_sessions";
  const std::string search_actions_table_name = "search_sessions_actions";
};
}  // namespace image_ranker