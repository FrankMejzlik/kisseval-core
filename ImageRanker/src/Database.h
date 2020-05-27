#pragma once

#include <string>
#include <vector>
#include "common.h"

#include <sqlite3.h>

namespace image_ranker
{
class Database
{
  /*
   * Methods
   */
 public:
  Database() = delete;
  Database(const Database& other) = delete;
  Database(Database&& other) = default;
  Database& operator=(const Database& other) = delete;
  Database& operator=(Database&& other) = default;
  ~Database() noexcept = default;

  Database(const std::string& db_filepath);

  size_t GetLastId() const;
  std::string GetErrorDescription() const;
  size_t GetErrorCode() const;

  size_t NoResultQuery(const std::string& query) const;
  std::pair<size_t, std::vector<std::vector<std::string>>> ResultQuery(const std::string& query) const;

  std::string EscapeString(const std::string& stringToEscape) const;

  /*
   * Member variables
   */
 private:
  /** Ptr to the database instance */
  std::unique_ptr<sqlite3, void (*)(sqlite3*)> _db;

  /** Filepath of source database file */
  std::string _db_fpth;
};

}  // namespace image_ranker
