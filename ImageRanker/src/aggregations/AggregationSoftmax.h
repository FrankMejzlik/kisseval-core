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
public:
  struct Settings
  {
    
  };

  // Methods
public:
  AggregationSoftmax() :
    AggregationFunctionBase(AggregationId::cSoftmax)
  {
  }

  virtual void SetAggregationSettings(AggregationSettings settingsString) override
  {

  }

  virtual bool CalculateTransformedVectors(const std::unordered_map<size_t, std::unique_ptr<Image>>& images) const
  {
    // Softmax file is loaded by default
    return true;
  }





  // Attributes
private:
  Settings _settings;

};