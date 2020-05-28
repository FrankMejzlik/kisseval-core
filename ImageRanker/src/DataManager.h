#pragma once

#include "common.h"

#include "Database.h"

namespace image_ranker
{
class ImageRanker;

/** Data representing one search session's action.
 * \see image_ranker::SearchSession
 */
struct SearchSessionAction
{
  size_t session_ID;
  size_t action_index;
  size_t rank;
};

/** Data representing one search session. */
struct SearchSession
{
  size_t ID;
  FrameId target_frame_ID;
  size_t duration;
  bool found;
  std::vector<SearchSessionAction> actions;
};

/**
 * Class responsible for manipulation with the data.
 *
 * It communicates with the database. Stores and loads data on demand and provides it to a caller.
 */
class DataManager
{
  friend class Tester;

  /*
   * Methods
   */
 public:
  // -----------------------------------------
  // No moving or copying.
  DataManager() = delete;
  DataManager(const DataManager& other) = delete;
  DataManager(DataManager&& other) = delete;
  DataManager& operator=(const DataManager& other) = delete;
  DataManager& operator=(DataManager&& other) = delete;
  ~DataManager() noexcept = default;
  // -----------------------------------------

  /** Main ctor */
  DataManager(ImageRanker* p_owner);

  void submit_annotator_user_queries(const StringId& data_pack_ID, const StringId& vocab_ID,
                                     const ::std::string& model_options, size_t user_level, bool with_example_images,
                                     const std::vector<AnnotatorUserQuery>& user_queries);

  [[nodiscard]] std::vector<UserTestQuery> fetch_user_test_queries(eUserQueryOrigin queries_origin,
                                                                   const StringId& vocabulary_ID,
                                                                   const StringId& data_pack_ID = ""s,
                                                                   const StringId& model_options = ""s) const;

  [[nodiscard]] std::vector<UserTestNativeQuery> fetch_user_native_test_queries(eUserQueryOrigin queries_origin) const;

  void submit_search_session(const std::string& data_pack_ID, const std::string& vocabulary_ID,
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
  [[nodiscard]] static QuantileLineChartData<size_t, float> get_aggregate_rank_progress_data(
      const std::vector<SearchSession>& sessions, size_t max_sess_len, size_t num_frames_total,
      size_t min_samples_count, bool normalize);

  [[nodiscard]] static MedianLineMultichartData<size_t, float> get_median_multichart_rank_progress_data(
      const std::vector<SearchSession>& sessions, size_t max_sess_len, size_t num_frames_total,
      size_t min_samples_count, bool normalize);

  [[nodiscard]] std::pair<std::vector<SearchSession>, std::vector<SearchSessionAction>> fetch_search_sessions(
      const std::string& data_pack_ID, size_t max_user_level) const;

  /*
   * Member variables
   */
 private:
  /** Pointer to this class's owner. */
  ImageRanker* _p_owner;

  /** Database for the user data. */
  Database _db;

  /** Table name where to store user queries. */
  const std::string queries_table_name = "user_queries";

  /** Table name where to store search sessions. */
  const std::string searches_table_name = "search_sessions";

  /** Table name where to store actions for search sessions. */
  const std::string search_actions_table_name = "search_sessions_actions";
};
}  // namespace image_ranker