#pragma once


#include "AggregationFunctionBase.h"



/*!
 * Softmax aggregation
 *
 * WARNING:
 *    Calling CalculateTransformedVectors() will load values from file.
 */
class AggregationSoftmax :
  public AggregationFunctionBase
{
  // Methods
public:
  AggregationSoftmax() :
    AggregationFunctionBase(AggregationId::cSoftmax)
  {
  }

  virtual bool CalculateTransformedVectors(const std::unordered_map<size_t, Image>& images) const
  {
    // Softmax file is loaded by default
  }

private:


  // Attributes
private:


};