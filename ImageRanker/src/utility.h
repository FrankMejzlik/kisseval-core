#pragma once


#include <vector>
#include <string>
#include <sstream>

using Clause = std::vector<size_t>;
using CnfFormula = std::vector<Clause>;

inline std::vector<std::string> SplitString(const std::string& s, char delimiter)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter))
  {
    tokens.push_back(token);
  }
  return tokens;
}

