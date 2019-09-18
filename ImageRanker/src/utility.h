#pragma once


#include <vector>
#include <string>
using namespace std::literals;
#include <sstream>
#include <random>
#include <charconv>
#include <iostream>

#include "common.h"


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


inline unsigned int FastAtoU(const char *str)
{
  unsigned int val = 0;
  while (*str) {
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

inline int strToInt(const std::string& str)
{
  int result;

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
    LOG_ERROR("Conversion of string '"s + str + "' failed with error code "s + std::to_string((int)ec) + ".");
  }
}