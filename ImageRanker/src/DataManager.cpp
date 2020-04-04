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

std::vector<GameSessionQueryResult> DataManager::submit_annotator_user_queries(
    const StringId& data_pack_ID, const StringId& vocab_ID, size_t user_level,
    const std::vector<AnnotatorUserQuery>& user_queries)
{
//<{user_query: }>,
//<{readable_user_query: }>,
//<{vocabulary_ID: }>,
//<{data_pack_ID: }>,
//<{model_options: }>,
//<{target_frame_ID: }>,
//<{with_example_images: 0}>,
//<{user_level: 0}>,
//<{manually_validated: 0}>,
//<{session_ID: }>,
//<{created: current_timestamp()}>);

std::string sqlQuery("INSERT INTO `" + db_name + "`.`" + queries_table_name +
                     "`"
                     "(`user_query`,`readable_user_query`,`vocabulary_ID`,`data_pack_ID`,`model_options`,`target_frame_"
                     "ID`,`with_example_images`,`user_level`,`manually_validated`,`session_ID`) VALUES ");
//
//for (auto&& query : user_queries)
//{
//  sqlQuery += "('"s + EncodeAndQuery(query.user_query_encoded) + "', '"s + query.user_query_readable + "'," +
//              +std::to_string(query.target_frame_ID) + ", "s + std::to_string((size_t)std::get<1>(data_ID)) + ", " +
//              std::to_string(imageId) + ", "s + std::to_string(originNumber) + ", '"s + std::get<0>(query) + "'),"s;
//}
//
//sqlQuery.pop_back();
//sqlQuery += ";";
//
//auto result = _db.NoResultQuery(sqlQuery);
//if (result != 0)
//{
//  LOG_ERROR(std::string("query: ") + sqlQuery + std::string("\n\nInserting queries into DB failed with error code: ") +
//            std::to_string(result));
//}

/******************************
  Construct result for user
*******************************/
std::vector<GameSessionQueryResult> userResult;
userResult.reserve(user_queries.size());
//
//for (auto&& query : inputQueries)
//{
//  // Get user keywords tokens
//  std::vector<std::string> userKeywords{StringenizeAndQuery(data_ID, std::get<2>(query))};
//
//  // Get image ID
//  size_t imageId = std::get<1>(query);
//
//  // Get image filename
//  std::string imageFilename{GetImageFilenameById(imageId)};
//
//  std::vector<std::pair<std::string, float>> netKeywordsProbs{};
//
//  userResult.emplace_back(std::get<0>(query), std::move(imageFilename), std::move(userKeywords),
//                          GetHighestProbKeywords(data_ID, imageId, 10ULL));
//}

return userResult;
}
