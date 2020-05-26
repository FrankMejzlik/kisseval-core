#pragma once

#include <string>
#include <vector>
#include "common.h"

#include <sqlite3.h>

namespace image_ranker
{
class Database
{
 public:
  Database() = delete;
  Database(const std::string& db_filepath);
  ~Database() noexcept;

  size_t GetLastId() const;
  std::string GetErrorDescription() const;
  size_t GetErrorCode() const;

  size_t NoResultQuery(const std::string& query) const;
  std::pair<size_t, std::vector<std::vector<std::string>>> ResultQuery(const std::string& query) const;

  std::string EscapeString(const std::string& stringToEscape) const;

 private:
  sqlite3* _db;

  std::string _db_fpth;
};

}  // namespace image_ranker
