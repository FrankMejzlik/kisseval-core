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
    TransformationFunctionBase(InputDataTransformId::cSoftmax)
  {
  }

  virtual void SetTransformationSettings(InputDataTransformSettings settingsString) override
  {

  }

  virtual bool CalculateTransformedVectors(const std::vector<std::unique_ptr<Image>>& images) const
  {
    // Softmax file is loaded by default
    return true;
  }

  virtual bool LowMem_CalculateTransformedVectors(const std::vector<std::unique_ptr<Image>>& images, size_t settings) const
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