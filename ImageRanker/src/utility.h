#pragma once

#include <string>
using namespace std::literals;

#include <array>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

#include "common.h"

using namespace image_ranker;

/**
 * Computes outer sum of discrete function defined by given chart data
 */
inline float calc_chart_area(const ModelTestResult& chart_data)
{
  float area{0.0F};
  uint32_t prev_x{0};
  for (auto&& [x, fx] : chart_data)
  {
    area += float((x - prev_x) * fx);

    prev_x = x;
  }

  return area;
}

inline json parse_data_config_file(const std::string& filepath)
{
  // read a JSON file
  std::ifstream i(filepath);
  json j;
  i >> j;

  return j;
}
/**
 * Parses string representation of tree CNF formula.
 *
 * EXAMPLE INPUT: &-20+--1+-3++-55+-333+
 */
inline CnfFormula parse_cnf_string(const std::string& string)
{
  std::vector<Clause> result;

  std::stringstream idx_ss;
  KeywordId idx_buffer;
  bool negate_next_atom{false};

  size_t depth{0};

  Clause clause_buffer;

  for (auto&& c : string)
  {
    if (bool(std::isdigit(int(c))))
    {
      idx_ss << c;
      continue;
    }

    // Flush index SS
    if (idx_ss.rdbuf()->in_avail() > 0)
    {
      idx_ss >> idx_buffer;
      clause_buffer.emplace_back(Literal<KeywordId>{idx_buffer, negate_next_atom});
      idx_ss = std::stringstream();
    }

    if (c == '&' || c == '|')
    {
      continue;
    }

    if (c == '-')
    {
      ++depth;
      continue;
    }

    if (c == '+')
    {
      --depth;
      if (depth == 0)
      {
        // Dispatch clause
        result.emplace_back(clause_buffer);
        clause_buffer = Clause();
      }
      continue;
    }

    if (c == '~')
    {
      negate_next_atom = true;
      continue;
    }
  }

  return result;
}

inline std::vector<std::string> split(const std::string& str, char delim)
{
  std::vector<std::string> result;
  std::stringstream ss(str);
  std::string item;

  while (getline(ss, item, delim))
  {
    result.emplace_back(item);
  }

  return result;
}

inline std::vector<std::string> split(const std::string& str, const std::string& delim)
{
  std::vector<std::string> result;

  std::string s{str};

  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delim)) != std::string::npos)
  {
    token = s.substr(0, pos);
    result.emplace_back(token);
    s.erase(0, pos + delim.length());
  }
  result.emplace_back(s);

  return result;
}

/***********************************
***************************************/

inline std::array<char, 4> floatToBytesLE(float number)
{
  std::array<char, 4> byteArray;

  char* bitNumber{reinterpret_cast<char*>(&number)};

  std::get<0>(byteArray) = bitNumber[0];
  std::get<1>(byteArray) = bitNumber[1];
  std::get<2>(byteArray) = bitNumber[2];
  std::get<3>(byteArray) = bitNumber[3];

  return byteArray;
}

inline std::array<char, 4> uint32ToBytesLE(uint32_t number)
{
  std::array<char, 4> byteArray;

  char* bitNumber{reinterpret_cast<char*>(&number)};

  std::get<0>(byteArray) = bitNumber[0];
  std::get<1>(byteArray) = bitNumber[1];
  std::get<2>(byteArray) = bitNumber[2];
  std::get<3>(byteArray) = bitNumber[3];

  return byteArray;
}

inline int32_t ParseIntegerLE(const std::byte* pFirstByte)
{
  // Initialize value
  int32_t signedInteger = 0;

  // Construct final BE integer
  signedInteger = static_cast<uint32_t>(pFirstByte[3]) << 24 | static_cast<uint32_t>(pFirstByte[2]) << 16 |
                  static_cast<uint32_t>(pFirstByte[1]) << 8 | static_cast<uint32_t>(pFirstByte[0]);

  // Return parsed integer
  return signedInteger;
}

inline float ParseFloatLE(const std::byte* pFirstByte)
{
  // Initialize temp value
  uint32_t byteFloat = 0;

  // Get correct unsigned value of float data
  byteFloat = static_cast<uint32_t>(pFirstByte[3]) << 24 | static_cast<uint32_t>(pFirstByte[2]) << 16 |
              static_cast<uint32_t>(pFirstByte[1]) << 8 | static_cast<uint32_t>(pFirstByte[0]);

  // Return reinterpreted data
  return *(reinterpret_cast<float*>(&byteFloat));
}

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

inline unsigned int FastAtoU(const char* str)
{
  unsigned int val = 0;
  while (*str)
  {
    val = (val << 1) + (val << 3) + *(str++) - 48;
  }
  return val;
}

/**
 * Returns ingeger sampled from uniform distribution from the interval [from, to].
 */
template <typename T>
inline T rand_integral(T from, T to)
{
  // Create random generator
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<T> randFromDistribution(from, to);

  return randFromDistribution(rng);
}

inline float strToFloat(const std::string& str)
{
  float result;

  std::stringstream ss{str};

  ss >> result;
  return result;

  /*
  // Convert and check if successful
  if (
    auto[p, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
    ec == std::errc()
    )
  {
    return result;
  }
  // If failed
  else
  {
    LOG_ERROR("Conversion of string '"s + str +"' failed with error code "s + std::to_string((int)ec) +".");
  }
  */
}
/**
 * Converts provided string into the `T` type.
 *
 * If not convertible we return defaultly constructed value of T.
 * Therefore we assume T to be default constructible.
 */
template <typename T>
inline T strTo(const std::string& str)
{
  T result;

  // Convert and check if successful
  if (auto [p, ec] = std::from_chars(str.data(), str.data() + str.size(), result); ec == std::errc())
  {
    return result;
  }
  // If failed
  else
  {
    LOG_ERROR("Conversion of string '"s + str + "' failed with error code "s + std::to_string((int)ec) + ".");
    return T();
  }
}

[[nodiscard]] inline std::vector<std::string> tokenize_query_and(std::string_view query)
{
  char separator_char = '&';

  // Prepare sstream for parsing from it
  std::stringstream query_ss(query.data());

  std::vector<std::string> result_tokens;
  {
    std::string token_buffer;
    while (std::getline(query_ss, token_buffer, separator_char))
    {
      if (token_buffer.empty())
      {
        continue;
      }

      // Push new token into the result
      result_tokens.emplace_back(std::move(token_buffer));
    }
  }

  return result_tokens;
}
[[nodiscard]] inline std::string encode_and_query(const std::string& query)
{
  auto word_IDs(tokenize_query_and(query));

  std::string result_encoded_query{"&"s};

  for (auto&& ID : word_IDs)
  {
    result_encoded_query += "-"s + ID + "+"s;
  }

  return result_encoded_query;
}