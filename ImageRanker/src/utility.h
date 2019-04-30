#pragma once


#include <vector>
#include <string>
#include <sstream>
#include <random>

using Clause = std::vector<std::pair<bool, size_t>>;
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