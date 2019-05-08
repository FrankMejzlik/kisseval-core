#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "common.h"
#include "utility.h"


class AggregationFunctionBase
{
  // Methods
public:
  AggregationFunctionBase(AggregationId id) :
    _id(id)
  {
  }

  virtual bool CalculateTransformedVectors(const std::unordered_map<size_t, Image>& images) const = 0;

  size_t GetId() const { return static_cast<size_t>(_id); }


  // Attributes
private:
  AggregationId _id;
};