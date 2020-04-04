#pragma once

#include <string>
#include <vector>
using namespace std::literals;
#include <array>
#include <charconv>
#include <iostream>
#include <random>
#include <sstream>

#include "common.h"

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

inline int GetRandomInteger(int from, int to)
{
  // Create random generator
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int> randFromDistribution(from, to);

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
[[nodiscard]] inline std::string EncodeAndQuery(const std::string& query)
{
  auto word_IDs(tokenize_query_and(query));

  std::string result_encoded_query{"&"s};

  for (auto&& ID : word_IDs)
  {
    result_encoded_query += "-"s + ID + "+"s;
  }

  return result_encoded_query;
}