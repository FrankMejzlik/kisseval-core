#pragma once

#include "common.h"

#include "Database.h"

namespace image_ranker
{
class ImageRanker;

class DataManager
{
 public:
  DataManager(ImageRanker* p_owner);

  void submit_annotator_user_queries(const StringId& data_pack_ID, const StringId& vocab_ID,
                                     const ::std::string& model_options, size_t user_level, bool with_example_images,
                                     const std::vector<AnnotatorUserQuery>& user_queries);

  std::vector<UserTestQuery> fetch_user_test_queries(eUserQueryOrigin queries_origin, const StringId& vocabulary_ID,
                                                     const StringId& data_pack_ID = ""s,
                                                     const StringId& model_options = ""s) const;

  std::vector<UserTestNativeQuery> fetch_user_native_test_queries(eUserQueryOrigin queries_origin) const;

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
  [[nodiscard]] QuantileLineChartData<size_t, float> get_search_sessions_rank_progress_chart_data(
      const std::string& data_pack_ID, const std::string& model_options, size_t max_user_level) const;

 private:
  ImageRanker* _p_owner;
  Database _db;

  const std::string db_name = PRIMARY_DB_DB_NAME;
  const std::string queries_table_name = "user_queries";
  const std::string searches_table_name = "search_sessions";
  const std::string search_actions_table_name = "search_sessions_actions";
};
}  // namespace image_ranker