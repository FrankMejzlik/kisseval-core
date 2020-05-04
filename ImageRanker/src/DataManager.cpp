#include "DataManager.h"

#include "utility.h"

#include "ImageRanker.h"

using namespace image_ranker;

DataManager::DataManager(ImageRanker* p_owner)
    : _p_owner(p_owner),
      _db(PRIMARY_DB_HOST, PRIMARY_DB_PORT, PRIMARY_DB_USERNAME, PRIMARY_DB_PASSWORD, PRIMARY_DB_DB_NAME)
{
  // Connect to the database
  auto result = _db.EstablishConnection();
  if (result != 0)
  {
    LOGE("Connecting to primary DB failed.");
  }
}

void DataManager::submit_annotator_user_queries(const StringId& data_pack_ID, const StringId& vocab_ID,
                                                const ::std::string& model_options, size_t user_level,
                                                bool with_example_images,
                                                const std::vector<AnnotatorUserQuery>& user_queries)
{
  std::stringstream sql_query_ss;
  sql_query_ss << ("INSERT INTO `" + db_name + "`.`" + queries_table_name +
                   "`"
                   "(`user_query`,`readable_user_query`,`vocabulary_ID`,`data_pack_ID`,`model_options`,`target_frame_"
                   "ID`,`with_example_images`,`user_level`,`manually_validated`,`session_ID`) VALUES ");

  std::string ex_imgs_str(with_example_images ? "1" : "0");

  /* \todo This is now taking only single query.
            We need to add temp query support. */
  size_t i = 0;
  for (auto&& query : user_queries)
  {
    sql_query_ss << "('"s << query.user_query_encoded.at(0) + "','"s << query.user_query_readable.at(0) << "','"
                 << vocab_ID << "','" << data_pack_ID << "','" + model_options + "',"
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
  auto result = _db.NoResultQuery(sql_query);
  if (result != 0)
  {
    LOGE("SQL query result: "s + sql_query + "\n\t Inserting queries into DB failed with error code: "s +
         std::to_string(result));
  }
}

std::vector<UserTestQuery> DataManager::fetch_user_test_queries(eUserQueryOrigin queries_origin,
                                                                const StringId& vocabulary_ID,
                                                                const StringId& data_pack_ID,
                                                                const StringId& model_options) const
{
  std::vector<UserTestQuery> result;

  std::stringstream SQL_query_ss;
  SQL_query_ss << "SELECT `target_frame_ID`, `user_query` FROM `" << _db.GetDbName() << "`.`" << queries_table_name
               << "` ";
  SQL_query_ss << "WHERE (`user_level` = " << int(queries_origin) << " AND `vocabulary_ID` = '" << vocabulary_ID
               << "' ";

  if (!data_pack_ID.empty())
  {
    SQL_query_ss << " AND `data_pack_ID` = '" << data_pack_ID << "'";
  }
  if (!model_options.empty())
  {
    SQL_query_ss << " AND `model_options` = '" << model_options << "'";
  }

  SQL_query_ss << ");";

  auto [res, db_rows] = _db.ResultQuery(SQL_query_ss.str());

  if (res != 0)
  {
    LOGE("Error fetching user queries from the DB. \n\n Error code: "s + std::to_string(res));
  }

  // Parse DB results
  for (auto&& row : db_rows)
  {
    FrameId target_frame_ID{ strTo<FrameId>(row[0]) };
    CnfFormula single_query{ parse_cnf_string(row[1]) };

    std::vector<CnfFormula> query;
    query.emplace_back(single_query);

    result.emplace_back(query, target_frame_ID);
  }

  return result;
}

std::vector<UserTestNativeQuery> DataManager::fetch_user_native_test_queries(eUserQueryOrigin queries_origin) const
{
  std::vector<UserTestNativeQuery> result;

  std::stringstream SQL_query_ss;
  SQL_query_ss << "SELECT `target_frame_ID`, `user_query` FROM `" << _db.GetDbName() << "`.`" << queries_table_name
               << "` ";
  SQL_query_ss << "WHERE (`user_level` = " << int(queries_origin) << " AND `vocabulary_ID` = 'native_language');";

  auto [res, db_rows] = _db.ResultQuery(SQL_query_ss.str());

  if (res != 0)
  {
    LOGE("Error fetching user queries from the DB. \n\n Error code: "s + std::to_string(res));
  }

  // Parse DB results
  for (auto&& row : db_rows)
  {
    FrameId target_frame_ID{ strTo<FrameId>(row[0]) };

    std::vector<std::string> query;
    query.emplace_back(row[1]);

    result.emplace_back(query, target_frame_ID);
  }

  return result;
}

bool DataManager::submit_search_session(const std::string& data_pack_ID, const std::string& vocabulary_ID,
                                        const std::string& model_commands, size_t user_level, bool with_example_images,
                                        FrameId target_frame_ID, eSearchSessionEndStatus end_status, size_t duration,
                                        const std::string& session_ID,
                                        const std::vector<InteractiveSearchAction>& actions)
{
  if (actions.empty())
  {
    return false;
  }

  /*
   * Insert into "search_sessions" table
   */
  std::stringstream query1Ss{ "" };
  query1Ss << "INSERT INTO `" << db_name << "`.`" << searches_table_name << "` ";
  query1Ss << "(`target_frame_ID`,`vocabulary_ID`,`data_pack_ID`,`model_options`,`duration`,`result`,";
  query1Ss << "`with_example_images`,`user_level`,`session_ID`,`manually_validated`) VALUES ";

  query1Ss << "(" << target_frame_ID << ",'" << vocabulary_ID << "','" << data_pack_ID << "','" << model_commands
           << "',";
  query1Ss << duration << "," << int(end_status) << "," << with_example_images << "," << user_level;
  query1Ss << ",'" << session_ID << "'," << int(false) << ");";

  // Run first query
  auto q1{ query1Ss.str() };
  size_t result1{ _db.NoResultQuery(q1) };
  if (result1 != 0)
  {
    LOGE("Failed to insert into:" + db_name + "`.`" + searches_table_name + "!");
  }
  // Currently inserted ID
  size_t id{ _db.GetLastId() };

  /*
   * Insert into "search_sessions_actions" table
   */
  std::stringstream query2Ss;

  query2Ss << "INSERT INTO `" << db_name << "`.`" << search_actions_table_name << "` ";
  query2Ss << "(`search_session_ID`,`action_index`,`action_query_index`,`action`,`operand`,`operand_readable`,`result_"
              "target_rank`,`time`) VALUES ";

  {
    size_t i{ 0_z };
    for (auto&& a : actions)
    {
      query2Ss << "(" << id << "," << i << "," << a.query_idx << ",'" << a.action << "','" << a.operand << "','"
               << a.operand_readable << "'," << a.final_rank << "," << a.time << ")";

      if (i < actions.size() - 1)
      {
        query2Ss << ",";
      }
      ++i;
    }
    query2Ss << ";";
  }

  auto q2{ query2Ss.str() };
  size_t result2{ _db.NoResultQuery(q2) };

  if (result1 != 0 || result2 != 0)
  {
    LOGE("Failed to insert into:" + db_name + "`.`" + search_actions_table_name + "!");
  }
}

QuantileLineChartData<size_t, float> DataManager::get_search_sessions_rank_progress_chart_data(
    const std::string& data_pack_ID, const std::string& model_options, size_t max_user_level) const
{
  LOGW("Not implemented");

  QuantileLineChartData<size_t, float> res;
  res.x = std::vector<size_t>{ 1, 2, 3, 4, 5 };
  res.y_min = std::vector<float>{ 200.0F, 250.0F, 100.0F, 60.0F, 20.0F };

  res.y_q1 = std::vector<float>{ 2000.0F, 1000.0F, 500.0F, 500.0F, 400.0F };
  res.y_q2 = std::vector<float>{ 4000.0F, 2000.0F, 1000.0F, 600.0F, 500.0F };
  res.y_q3 = std::vector<float>{ 8000.0F, 4000.0F, 3000.0F, 2000.0F, 800.0F };

  res.y_max = std::vector<float>{ 18000.0F, 14000.0F, 10000.0F, 6000.0F, 5000.0F };

  return res;
}