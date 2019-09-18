#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "common.h"

class Image
{
public:
  struct ScoringDataInfo
  {
    float m_min;
    float m_max;
    float m_mean;
    float m_variance;
  };

public:
  Image() = default;

  Image(
    size_t id,
    size_t index,
    const std::string& filename,
    size_t videoId, size_t shotId, size_t frameNumber
  ) :
    m_imageId(id),
    m_index(index),
    m_numSuccessorFrames(0_z),
    m_filename(std::move(filename)),
    m_videoId(videoId),
    m_shotId(shotId),
    m_frameNumber(frameNumber)
  {}

  std::unordered_map<TransformFullId, std::vector<float>>* GetScoringVectorsPtr(KwScoringDataId kwScDataId)
  {
    // If this kwsc ID found
    if (
      auto&& transformMapIt{ _transformedImageScoringData.find(kwScDataId) };
      transformMapIt != _transformedImageScoringData.end())
    {
      return &(transformMapIt->second);
    }
    else
    {
      return nullptr;
    }
  }

  const std::unordered_map<TransformFullId, std::vector<float>>* GetScoringVectorsConstPtr(KwScoringDataId kwScDataId) const
  {
    // If this kwsc ID found
    if (
      auto&& transformMapIt{ _transformedImageScoringData.find(kwScDataId) };
      transformMapIt != _transformedImageScoringData.end())
    {
      return &(transformMapIt->second);
    }
    else
    {
      return nullptr;
    }
  }

  size_t GetNumBins(KwScoringDataId kwScDataId) const 
  { 
    // If this kwsc ID found
    if (
      auto ptr{ GetScoringVectorsConstPtr(kwScDataId) };
      ptr != nullptr)
    {
      return ptr->size();
    }
    
    return SIZE_T_ERROR_VALUE;    
  }

  const std::vector<float>* GetAggregationVectorById(KwScoringDataId kwScDataId, size_t transformId) const
  {
    // If found
    if (
      auto ptr{ GetScoringVectorsConstPtr(kwScDataId) };
      ptr != nullptr)
    {
      if (
        auto&& it{ ptr->find(transformId) };
        it != ptr->end())
      {
        return &(it->second);
      }
      else
      {
        LOG_ERROR("Transformation not found!");
        return nullptr;
      }
    }
    else 
    {
      LOG_ERROR("KwScoringData ID not found!");
      return nullptr;
    }
  }
 


  size_t m_imageId;
  size_t m_index;
  size_t m_numSuccessorFrames;
  std::string m_filename;

  size_t m_videoId;
  size_t m_shotId;
  size_t m_frameNumber;


  //! Data clone for user simulation
  std::map <KwScoringDataId, std::vector<float>> _rawSimUserData;

  //! Top ranked keywords for this image
  std::map<KwScoringDataId, KeywordPtrScoringPair> _topKeywords;

  //! Raw input data
  std::map<KwScoringDataId, std::vector<float>> _rawImageScoringData;
  std::map<KwScoringDataId, ScoringDataInfo> _rawImageScoringDataInfo;

  //! Aggregation vectors calculated by provided aggregations
  std::map<KwScoringDataId, std::unordered_map<TransformFullId, std::vector<float>>> _transformedImageScoringData;
};
