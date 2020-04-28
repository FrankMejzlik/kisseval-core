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

 private:
 private:
  ImageRanker* _p_owner;
  Database _db;

  const std::string db_name = PRIMARY_DB_DB_NAME;
  const std::string queries_table_name = "user_queries";
  const std::string searches_table_name = "search_sessions";
  const std::string search_actions_table_name = "search_sessions_actions";
};
}  // namespace image_ranker