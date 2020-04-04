#include "DataManager.h"

#include "utility.h"

#include "ImageRanker.h"

DataManager::DataManager(ImageRanker* p_owner)
    : _p_owner(p_owner),
      _db(PRIMARY_DB_HOST, PRIMARY_DB_PORT, PRIMARY_DB_USERNAME, PRIMARY_DB_PASSWORD, PRIMARY_DB_DB_NAME)
{
  // Connect to the database
  auto result = _db.EstablishConnection();
  if (result != 0)
  {
    LOG_ERROR("Connecting to primary DB failed.");
  }
}

void DataManager::submit_annotator_user_queries(
    const StringId& data_pack_ID, const StringId& vocab_ID, size_t user_level, bool with_example_images,
    const std::vector<AnnotatorUserQuery>& user_queries)
{
  std::stringstream sql_query_ss;
  sql_query_ss << ("INSERT INTO `" + db_name + "`.`" + queries_table_name +
                   "`"
                   "(`user_query`,`readable_user_query`,`vocabulary_ID`,`data_pack_ID`,`model_options`,`target_frame_"
                   "ID`,`with_example_images`,`user_level`,`manually_validated`,`session_ID`) VALUES ");

  std::string ex_imgs_str(with_example_images ? "1" : "0");

  size_t i = 0;
  for (auto&& query : user_queries)
  {
    sql_query_ss << "('"s << encode_and_query(query.user_query_encoded) + "','"s << query.user_query_readable << "','"
                 << vocab_ID << "','" << data_pack_ID << "',NULL," << std::to_string(query.target_frame_ID) << ", "s
                 << ex_imgs_str << "," << std::to_string(user_level) << ",0,'" << query.session_ID + "')";

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
    LOG_ERROR("SQL query result: "s + sql_query + "\n\t Inserting queries into DB failed with error code: "s +
              std::to_string(result));
  }

}
