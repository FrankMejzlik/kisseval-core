#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"

class Image {
 public:
  struct ScoringDataInfo {
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
      size_t videoId, size_t shotId, size_t frameNumber) : m_imageId(id),
                                                           m_index(index),
                                                           m_numSuccessorFrames(0_z),
                                                           m_filename(std::move(filename)),
                                                           m_videoId(videoId),
                                                           m_shotId(shotId),
                                                           m_frameNumber(frameNumber) {}

  std::unordered_map<TransformFullId, std::vector<float>>* GetScoringVectorsPtr(DataId data_ID) {
    // If this kwsc ID found
    if (
        auto&& transformMapIt{_transformedImageScoringData.find(data_ID)};
        transformMapIt != _transformedImageScoringData.end()) {
      return &(transformMapIt->second);
    } else {
      return nullptr;
    }
  }

  const std::unordered_map<TransformFullId, std::vector<float>>* GetScoringVectorsConstPtr(DataId data_ID) const {
    // If this kwsc ID found
    if (
        auto&& transformMapIt{_transformedImageScoringData.find(data_ID)};
        transformMapIt != _transformedImageScoringData.end()) {
      return &(transformMapIt->second);
    } else {
      return nullptr;
    }
  }

  size_t GetNumBins(DataId data_ID) const {
    // If this kwsc ID found
    if (
        auto ptr{GetScoringVectorsConstPtr(data_ID)};
        ptr != nullptr) {
      return ptr->size();
    }

    return SIZE_T_ERROR_VALUE;
  }

  const std::vector<float>* GetAggregationVectorById(DataId data_ID, size_t transformId) const {
    // If no transform
    if (transformId == NO_TRANSFORM_ID) {
      auto i = _rawImageScoringData.find(data_ID);

      if (i == _rawImageScoringData.end()) {
        LOG_ERROR("KsSc ID not found!");
      }

      return &(i->second);
    }

    // If found
    if (
        auto ptr{GetScoringVectorsConstPtr(data_ID)};
        ptr != nullptr) {
      if (
          auto&& it{ptr->find(transformId)};
          it != ptr->end()) {
        return &(it->second);
      } else {
        LOG_ERROR("Transformation not found!");
        return nullptr;
      }
    } else {
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
  std::map<DataId, std::vector<float>> _rawSimUserData;

  //! Top ranked keywords for this image
  std::map<DataId, KeywordPtrScoringPair> _topKeywords;

  //! Raw input data
  std::map<DataId, std::vector<float>> _rawImageScoringData;
  std::map<DataId, ScoringDataInfo> _rawImageScoringDataInfo;

  //! Aggregation vectors calculated by provided aggregations
  std::map<DataId, std::unordered_map<TransformFullId, std::vector<float>>> _transformedImageScoringData;
};
