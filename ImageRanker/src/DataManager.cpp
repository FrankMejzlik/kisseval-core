#include "DataManager.h"

#include "utility.h"

#include "ImageRanker.h"

using namespace image_ranker;

DataManager::DataManager() : _db(DB_FPTH) {}

void DataManager::submit_annotator_user_queries(const StringId& data_pack_ID, const StringId& vocab_ID,
                                                const StringId& imageset_ID, const ::std::string& model_options,
                                                size_t user_level, bool with_example_images,
                                                const std::vector<AnnotatorUserQuery>& user_queries)
{
  std::stringstream sql_query_ss;
  sql_query_ss << ("INSERT INTO `" + queries_table_name +
                   "`"
                   "(`user_query`,`readable_user_query`,`vocabulary_ID`,`data_pack_ID`,`imageset_ID`,`model_options`,`"
                   "target_frame_"
                   "ID`,`with_example_images`,`user_level`,`manually_validated`,`session_ID`) VALUES ");

  std::string ex_imgs_str{ "0" };
  if (with_example_images)
  {
    ex_imgs_str = "1";
  }

  /* \todo This is now taking only simple query. We need to add temp query support. */
  size_t i = 0;
  for (auto&& query : user_queries)
  {
    sql_query_ss << "('"s << Database::escape_str(query.user_query_encoded.at(0)) + "','"s
                 << Database::escape_str(query.user_query_readable.at(0)) << "','" << vocab_ID << "','" << data_pack_ID
                 << "','" + imageset_ID + "','" + model_options + "',"
                 << std::to_string(query.target_sequence_IDs.at(0)) << ", "s << ex_imgs_str << ","
                 << std::to_string(user_level) << ",0,'" << query.session_ID + "')";

    if (i < (user_queries.size() - 1))
    {
      sql_query_ss << ",";
    }
    else
    {
      sql_query_ss << ";";
    }
    ++i;
  }

  std::string sql_query(sql_query_ss.str());

  // Run SQL query
  _db.no_result_query(sql_query);
}

std::vector<UserTestQuery> DataManager::fetch_user_test_queries(eUserQueryOrigin queries_origin,
                                                                const StringId& vocabulary_ID,
                                                                const StringId& data_pack_ID,
                                                                const StringId& model_options) const
{
  std::vector<UserTestQuery> result;

  std::stringstream SQL_query_ss;
  SQL_query_ss << "SELECT `target_frame_ID`, `user_query` FROM `" << queries_table_name << "` ";
  SQL_query_ss << "WHERE (`user_level` = " << int(queries_origin) << " AND `vocabulary_ID` = '" << vocabulary_ID << "'";

  if (false)
  //if (!data_pack_ID.empty())
  {
    SQL_query_ss << " AND `data_pack_ID` = '" << data_pack_ID << "'";
  }
  //if (!model_options.empty())
  if (false)
  {
    SQL_query_ss << " AND `model_options` = '" << model_options << "'";
  }

  SQL_query_ss << ");";

  auto str{ SQL_query_ss.str() };

  auto db_rows = _db.result_query(SQL_query_ss.str());

  // Parse DB results
  for (auto&& row : db_rows)
  {
    FrameId target_frame_ID{ str_to<FrameId>(row[0]) };
    CnfFormula single_query{ parse_cnf_string(row[1]) };

    std::vector<CnfFormula> query;
    query.emplace_back(single_query);

    result.emplace_back(query, target_frame_ID);
  }

  return result;
}

std::vector<UserTestNativeQuery> DataManager::fetch_user_native_test_queries(eUserQueryOrigin queries_origin,
                                                                             const std::string& imageset_ID) const
{
  std::vector<UserTestNativeQuery> result;

  std::stringstream SQL_query_ss;
  SQL_query_ss << "SELECT `target_frame_ID`, `user_query` FROM `" << queries_table_name << "` ";
  SQL_query_ss << "WHERE (`user_level` = " << int(queries_origin)
               << " AND `vocabulary_ID` = 'native_language' AND `imageset_ID` = '" << imageset_ID << "');";

  auto db_rows = _db.result_query(SQL_query_ss.str());
  auto str{ SQL_query_ss.str() };

  std::cout << str << std::endl;
  // Parse DB results
  for (auto&& row : db_rows)
  {
    FrameId target_frame_ID{ str_to<FrameId>(row[0]) };

    std::vector<std::string> query;
    query.emplace_back(row[1]);

    result.emplace_back(query, target_frame_ID);
  }

  return result;
}

void DataManager::submit_search_session(const std::string& data_pack_ID, const std::string& vocabulary_ID,
                                        const std::string& model_options, size_t user_level, bool with_example_images,
                                        FrameId target_frame_ID, eSearchSessionEndStatus end_status, size_t duration,
                                        const std::string& session_ID,
                                        const std::vector<InteractiveSearchAction>& actions)
{
  // If nothing to do
  if (actions.empty())
  {
    LOGW("Submitting empty search session. Ignoring it.");
    return;
  }

  /*
   * Insert into "search_sessions" table
   */
  std::stringstream query1Ss{ "" };
  query1Ss << "INSERT INTO `" << searches_table_name << "` ";
  query1Ss << "(`target_frame_ID`,`vocabulary_ID`,`data_pack_ID`,`model_options`,`duration`,`result`,";
  query1Ss << "`with_example_images`,`user_level`,`session_ID`,`manually_validated`) VALUES ";

  query1Ss << "(" << target_frame_ID << ",'" << vocabulary_ID << "','" << data_pack_ID << "','" << model_options
           << "',";
  query1Ss << duration << "," << int(end_status) << "," << with_example_images << "," << user_level;
  query1Ss << ",'" << session_ID << "'," << int(false) << ");";

  // Run first query
  auto q1{ query1Ss.str() };

  _db.no_result_query(q1);

  // Currently inserted ID
  size_t id{ _db.get_last_inserted_ID() };

  /*
   * Insert into "search_sessions_actions" table
   */
  std::stringstream query2Ss;

  query2Ss << "INSERT INTO `" << search_actions_table_name << "` ";
  query2Ss << "(`search_session_ID`,`action_index`,`action_query_index`,`action`,`operand`,`operand_readable`,`result_"
              "target_rank`,`time`,`is_initial`) VALUES ";

  {
    size_t i{ 0_z };
    for (auto&& a : actions)
    {
      query2Ss << "(" << id << "," << i << "," << a.query_idx << ",'" << a.action << "','" << a.operand << "','"
               << a.operand_readable << "'," << a.final_rank << "," << a.time << "," << size_t(a.is_initial) << ")";

      if (i < actions.size() - 1)
      {
        query2Ss << ",";
      }
      ++i;
    }
    query2Ss << ";";
  }

  auto q2{ query2Ss.str() };
  _db.no_result_query(q2);
}

SearchSessRankChartData DataManager::get_search_sessions_rank_progress_chart_data(
    const std::string& data_pack_ID, [[maybe_unused]] const std::string& model_options, size_t max_user_level,
    size_t num_frames, size_t min_samples_count, bool normalize) const
{
  // Get data for all sessions
  auto [sessions, all_actions]{ fetch_search_sessions(data_pack_ID, max_user_level) };

  size_t max_session_len{ 0 };

  // Add actions to all loaded sessions
  for (auto&& [ID, target_frame_ID, duration, found, actions] : sessions)
  {
    size_t counter{ 0 };
    for (auto&& [sess_ID, action_index, rank] : all_actions)
    {
      if (sess_ID != ID)
      {
        continue;
      }
      actions.emplace_back(SearchSessionAction{ sess_ID, action_index, rank });
      ++counter;
    }

    max_session_len = std::max(max_session_len, counter);
  }

  std::vector<size_t> counter;
  counter.resize(max_session_len + 1);

  // Filter out sessions of length we dont have enough samples from
  {
    size_t i{ 0 };
    for (auto&& [ID, target_frame_ID, duration, found, actions] : sessions)
    {
      size_t num_actions{ actions.size() };

      ++counter[num_actions];

      ++i;
    }
  }

  std::vector<SearchSession> sessions_filtered;
  size_t max_filtered_len{ 0_z };
  for (auto&& sess : sessions)
  {
    size_t num_actions{ sess.actions.size() };

    if (counter[num_actions] < min_samples_count)
    {
      continue;
    }

    max_filtered_len = std::max(max_filtered_len, num_actions);
    sessions_filtered.emplace_back(std::move(sess));
  }

  SearchSessRankChartData total_res;
  total_res.aggregate_quantile_chart =
      get_aggregate_rank_progress_data(sessions_filtered, max_filtered_len, num_frames, min_samples_count, normalize);
  total_res.median_multichart = get_median_multichart_rank_progress_data(sessions_filtered, max_filtered_len,
                                                                         num_frames, min_samples_count, normalize);

  return total_res;
}

QuantileLineChartData<size_t, float> DataManager::get_aggregate_rank_progress_data(
    const std::vector<SearchSession>& sessions, size_t max_sess_len, size_t num_frames_total, size_t min_samples_count,
    bool normalize)
{
  QuantileLineChartData<size_t, float> res;

  // Vector of vectors for all session lengths
  std::vector<std::vector<size_t>> sess_ranks(max_sess_len + 1, std::vector<size_t>{});
  for (auto&& [ID, target_frame_ID, duration, found, actions] : sessions)
  {
    size_t len{ 1_z };
    for (auto&& [sess_ID, action_index, rank] : actions)
    {
      sess_ranks[len].emplace_back(rank);
      ++len;
    }
  }

  // Sort those vectors
  size_t i{ 0_z };
  for (auto&& rank_vec : sess_ranks)
  {
    if (i == 0)
    {
      ++i;
      continue;
    }

    if (rank_vec.size() < min_samples_count)
    {
      continue;
    }

    std::sort(rank_vec.begin(), rank_vec.end());

    size_t count{ rank_vec.size() };

    if (count == 0)
    {
      continue;
    }

    size_t q1_idx{ static_cast<size_t>(round(count * 0.25F)) };  // NOLINT
    size_t q2_idx{ static_cast<size_t>(round(count * 0.5F)) };   // NOLINT
    size_t q3_idx{ static_cast<size_t>(round(count * 0.75F)) };  // NOLINT

    res.x.emplace_back(i);
    if (normalize)
    {
      res.y_min.emplace_back(float(rank_vec.front()) / num_frames_total);
      res.y_q1.emplace_back(float(rank_vec[q1_idx]) / num_frames_total);
      res.y_q2.emplace_back(float(rank_vec[q2_idx]) / num_frames_total);
      res.y_q3.emplace_back(float(rank_vec[q3_idx]) / num_frames_total);
      res.y_max.emplace_back(float(rank_vec.back()) / num_frames_total);
    }
    else
    {
      res.y_min.emplace_back(float(rank_vec.front()));
      res.y_q1.emplace_back(float(rank_vec[q1_idx]));
      res.y_q2.emplace_back(float(rank_vec[q2_idx]));
      res.y_q3.emplace_back(float(rank_vec[q3_idx]));
      res.y_max.emplace_back(float(rank_vec.back()));
    }
    res.count.emplace_back(count);

    ++i;
  }
  return res;
}

MedianLineMultichartData<size_t, float> DataManager::get_median_multichart_rank_progress_data(
    const std::vector<SearchSession>& sessions, size_t max_sess_len, size_t num_frames_total, size_t min_samples_count,
    bool normalize)
{
  // Populate rank values into specified vectors
  std::vector<std::vector<std::vector<size_t>>> sess_ranks(max_sess_len + 1);
  {
    size_t i{ 0_z };
    for (auto&& vec : sess_ranks)
    {
      vec.resize(i);
      ++i;
    }
  }
  for (auto&& [ID, target_frame_ID, duration, found, actions] : sessions)
  {
    size_t sess_len{ actions.size() };

    auto& vec{ sess_ranks[sess_len] };

    size_t i{ 0_z };
    for (auto&& [sess_ID, action_index, rank] : actions)
    {
      vec[i].emplace_back(rank);
      ++i;
    }
  }

  std::vector<std::vector<size_t>> x;
  std::vector<std::vector<float>> medians;

  // Do the counting
  for (auto&& vec_len_N : sess_ranks)
  {
    size_t i{ 0_z };
    std::vector<float> medians_for_len_N;
    std::vector<size_t> xs_for_len_N;
    for (auto&& sess_len_N_act_i : vec_len_N)
    {
      std::sort(sess_len_N_act_i.begin(), sess_len_N_act_i.end());

      xs_for_len_N.emplace_back(i + 1);

      size_t num_samples{ sess_len_N_act_i.size() };
      if (num_samples < min_samples_count)
      {
        continue;
      }

      // Get median index
      size_t median_idx{ size_t(round(num_samples / 2.0F)) };  // NOLINT
      float median = float(sess_len_N_act_i[median_idx]);

      if (normalize)
      {
        median /= num_frames_total;
      }

      medians_for_len_N.emplace_back(median);

      ++i;
    }

    if (!medians_for_len_N.empty())
    {
      x.emplace_back(xs_for_len_N);
      medians.emplace_back(medians_for_len_N);
    }
  }

  return MedianLineMultichartData<size_t, float>{ x, medians };
}

std::pair<std::vector<SearchSession>, std::vector<SearchSessionAction>> DataManager::fetch_search_sessions(
    const std::string& data_pack_ID, size_t max_user_level) const
{
  std::stringstream SQL_query_sessions_ss;
  SQL_query_sessions_ss << "SELECT `ID`, `target_frame_ID`, `duration`, `result` FROM `" << searches_table_name
                        << "` WHERE `data_pack_ID` = '" << data_pack_ID << "' AND `user_level` <= " << max_user_level
                        << ";";
  std::stringstream SQL_query_actions_ss;
  SQL_query_actions_ss << "SELECT `search_session_ID`, `action_index`, `result_target_rank` FROM `"
                       << search_actions_table_name << "` WHERE `action_query_index` = 0  AND `is_initial` = 0;";

  auto rows_sessions = _db.result_query(SQL_query_sessions_ss.str());
  auto rows_actions = _db.result_query(SQL_query_actions_ss.str());

  // Create all sessions
  std::vector<SearchSession> sessions;
  sessions.reserve(rows_sessions.size());
  for (auto&& row : rows_sessions)
  {
    size_t ID{ str_to<size_t>(row[0]) };
    FrameId target_frame_ID{ str_to<FrameId>(row[1]) };
    size_t duration{ str_to<size_t>(row[2]) };
    bool found{ static_cast<bool>(str_to<unsigned char>(row[3])) };

    sessions.emplace_back(SearchSession{ ID, target_frame_ID, duration, found, std::vector<SearchSessionAction>{} });
  }

  // Parse all actions
  std::vector<SearchSessionAction> all_actions;
  all_actions.reserve(rows_actions.size());
  for (auto&& row : rows_actions)
  {
    size_t search_session_ID{ str_to<size_t>(row[0]) };
    size_t action_index{ str_to<size_t>(row[1]) };
    size_t result_target_rank{ str_to<size_t>(row[2]) };

    all_actions.emplace_back(SearchSessionAction{ search_session_ID, action_index, result_target_rank });
  }

  return std::pair(std::move(sessions), std::move(all_actions));
}