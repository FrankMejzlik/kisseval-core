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