#pragma once


#include "TransformationFunctionBase.h"



/*!
 * Softmax aggregation
 *
 * WARNING:
 *    Calling CalculateTransformedVectors() will load values from file.
 */
class TransformationSoftmax :
  public TransformationFunctionBase
{
public:
  struct Settings
  {
    
  };

  // Methods
public:
  TransformationSoftmax() :
    TransformationFunctionBase(NetDataTransformation::cSoftmax)
  {
  }

  virtual void SetTransformationSettings(NetDataTransformSettings settingsString) override
  {

  }

  virtual bool CalculateTransformedVectors(const std::unordered_map<size_t, std::unique_ptr<Image>>& images) const
  {
    // Softmax file is loaded by default
    return true;
  }


  virtual size_t GetGuidFromSettings() const override
  {
    return GetGuid(0ULL);
  }


  // Attributes
private:
  Settings _settings;

};