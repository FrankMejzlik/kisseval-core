#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "common.h"

struct Image
{
  Image() = default;

  Image(
    size_t id,
    std::string&& filename,
    std::vector<float>&& rawNetRanking,
    float min, float max,
    float mean, float variance
  ) :
    m_imageId(id),
    m_numSuccessorFrames(0_z),
    m_filename(std::move(filename)),
    m_rawNetRanking(std::move(rawNetRanking)),
    m_rawNetRankingSorted(),
    m_min(min), m_max(max),
    m_mean(mean), m_variance(variance)
  {

    // Create sorted array
    size_t i{ 0ULL };
    for (auto&& img : m_rawNetRanking)
    {
      m_rawNetRankingSorted.emplace_back(std::pair(static_cast<uint32_t>(i), img));

      ++i;
    }


    // Sort it
    std::sort(
      m_rawNetRankingSorted.begin(), m_rawNetRankingSorted.end(),
      [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) -> bool
      {
        return a.second > b.second;
      }
    );


  }

  size_t GetNumBins() const { return m_rawNetRanking.size(); }
  const std::vector<float>* GetAggregationVectorById(size_t id) const
  {
    // If found
    if (
      auto&& it{ m_aggVectors.find(id) }; 
      it != m_aggVectors.end()
    )
    {
      return &(it->second);
    }   
    else 
    {
      LOG_ERROR("Aggregation not found!");
      return nullptr;
    }
  }


  size_t m_imageId;
  size_t m_numSuccessorFrames;
  std::string m_filename;

  //! Raw vector as it came out of neural network
  std::vector<float> m_rawNetRanking;

  float m_min;
  float m_max;
  float m_mean;
  float m_variance;

  //! Raw vector as it came out of neural network but SORTED
  std::vector<std::pair<uint32_t, float>> m_rawNetRankingSorted;
  std::vector<std::pair<size_t, float>> m_hypernymsRankingSorted;

  //! Softmax probability ranking
  std::vector<float> m_softmaxVector;
  std::vector<float> m_linearVector;

  //! Probability vector from custom MinMax Clamp method
  std::vector<float> m_minMaxLinearVector;
  
  


  //! Aggregation vectors calculated by provided aggregations
  std::unordered_map<size_t, AggregationVector> m_aggVectors;
};
