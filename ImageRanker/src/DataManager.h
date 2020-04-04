#pragma once

#include "common.h"

#include "Database.h"

class ImageRanker;

class DataManager
{
 public:
  DataManager(ImageRanker* p_owner);

  std::vector<GameSessionQueryResult> submit_annotator_user_queries(
      const StringId& data_pack_ID, const StringId& vocab_ID, size_t user_level, bool with_example_images,
      const std::vector<AnnotatorUserQuery>& user_queries);

 private:
 private:
  ImageRanker* _p_owner;
  Database _db;

  const std::string db_name = PRIMARY_DB_DB_NAME;
  const std::string queries_table_name = "user_queries";
  const std::string searches_table_name = "search_sessions";
  const std::string search_actions_table_name = "search_sessions_actions";
};
