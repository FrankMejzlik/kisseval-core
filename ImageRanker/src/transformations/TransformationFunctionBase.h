#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "common.h"
#include "utility.h"

class Image;

class TransformationFunctionBase
{
  // Methods
public:
  TransformationFunctionBase(InputDataTransformId id) :
    _id(id)
  {
  }

  // Required API for ranking models
public:
  virtual void SetTransformationSettings(InputDataTransformSettings settingsString) = 0;
  virtual bool CalculateTransformedVectors(const std::vector<std::unique_ptr<Image>>& images) const = 0;

  /*!
   * Calculates transformed data vector with low memory usage
   * 
   * NOTE:
   *  It will destroy source data vector
   * 
   * \param images
   * \return 
   */
  virtual bool LowMem_CalculateTransformedVectors(const std::vector<std::unique_ptr<Image>>& images, size_t settings) const = 0;
  virtual size_t GetGuidFromSettings() const = 0;

  // Methods
public:
  size_t GetId() const { return static_cast<size_t>(_id); }

  size_t GetGuid(size_t index) const
  {
    // GUID starts with Aggregation ID and then two decimal places for subIDs
    size_t guid{ GetId() + index };

    return guid;
  }
  


  // Attributes
private:
  InputDataTransformId _id;
};