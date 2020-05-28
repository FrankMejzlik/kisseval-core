
#include "Database.h"

#include "utility.h"

using namespace image_ranker;

std::string Database::escape_str(const std::string& raw_str)
{
  char* c_str{ sqlite3_mprintf("%q", raw_str.c_str()) };  // NOLINT
  return std::string{ c_str };
}

Database::Database(const std::string& db_filepath)
    : _db_fpth(db_filepath),
      // Add custom deleter for this pointer
      _p_db(nullptr, [](sqlite3* ptr) { sqlite3_close(ptr); })
{
  int rc{ ERR_VAL<int>() };

  // Instantiate database instance
  sqlite3* p_db{ nullptr };
  rc = sqlite3_open(_db_fpth.c_str(), &p_db);
  _p_db.reset(p_db);

  // Check for errors
  if (rc > 0)
  {
    std::string msg{ "Can't open database: " + std::string{ sqlite3_errmsg(_p_db.get()) } };
    LOGE(msg);
    PROD_THROW("An error occured!");
  }
};

std::string Database::get_last_error_msg() const { return std::string{ sqlite3_errmsg(_p_db.get()) }; }

size_t Database::get_last_err_code() const { return static_cast<size_t>(sqlite3_errcode(_p_db.get())); }

void Database::no_result_query(const std::string& query) const
{
  sqlite3_stmt* stmt;

  // Compile SQL statement
  if (sqlite3_prepare_v2(_p_db.get(), query.c_str(), -1, &stmt, NULL) != SQLITE_OK)
  {
    sqlite3_finalize(stmt);
    std::string msg{ "Error while compiling SQL with message '" + get_last_error_msg() + "'\n\t Query: " + query };
    LOGE(msg);
    PROD_THROW("An error occured!");
  }
  scope_exit stmt_guard([stmt]() { sqlite3_finalize(stmt); });

  int ret_code{ sqlite3_step(stmt) };
  if (ret_code != SQLITE_DONE)
  {
    std::string msg{ "Error while stepping SQL with message '" + get_last_error_msg() + "'\n\t Query: " + query };
    LOGE(msg);
    PROD_THROW("An error occured!");
  }
}

size_t Database::get_last_inserted_ID() const { return static_cast<size_t>(sqlite3_last_insert_rowid(_p_db.get())); }

std::vector<std::vector<std::string>> Database::result_query(const std::string& query) const
{
  // Send query to DB and get result
  std::vector<std::vector<std::string>> result;

  sqlite3_stmt* stmt;

  // Compile SQL statement
  if (sqlite3_prepare_v2(_p_db.get(), query.c_str(), -1, &stmt, NULL) != SQLITE_OK)
  {
    std::string msg{ "Error while compiling SQL with message '" + get_last_error_msg() + "'\n\t Query: " + query };
    LOGE(msg);
    PROD_THROW("An error occured!");
  }
  scope_exit stmt_guard([stmt]() { sqlite3_finalize(stmt); });

  int ret_code{ 0 };
  while ((ret_code = sqlite3_step(stmt)) == SQLITE_ROW)
  {
    // execute sql statement, and while there are rows returned, print ID
    auto cnt = sqlite3_data_count(stmt);

    std::vector<std::string> r;
    for (size_t i{ 0_z }; i < cnt; ++i)
    {
      const char* c = reinterpret_cast<const char*>(sqlite3_column_text(stmt, static_cast<int>(i)));  // NOLINT
      r.emplace_back(std::string(c));
    }
    result.emplace_back(std::move(r));
  }
  if (ret_code != SQLITE_DONE)
  {
    std::string msg{ "Error while stepping in SQL with message '" + get_last_error_msg() + "'\n\t Query: " + query };
    LOGE(msg);
    PROD_THROW("An error occured!");
  }

  return result;
}
