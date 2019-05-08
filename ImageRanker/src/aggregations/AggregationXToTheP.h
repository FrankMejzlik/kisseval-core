#pragma once

#include "AggregationFunctionBase.h"

/*!
 * Aggregation of type f(x) = x^p
 */
class AggregationXToTheP :
  public AggregationFunctionBase
{
public:
  struct Settings
  {
    float m_exponent;
  };


  // Methods
public:
  AggregationXToTheP() :
    AggregationFunctionBase(AggregationId::cXToTheP),
    _exponents({ 0.2f, 0.5f, 0.8f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 9.0f })
  {
  }

  virtual bool CalculateTransformedVectors(const std::unordered_map<size_t, Image>& images) const
  {

  }

  Settings GetDefaultSettings() const
  {
    // Return default settings instance
    return Settings{ 1.0f };
  }

  Settings DecodeModelSettings(ModelSettings settingsString) const
  {
    auto settings{ GetDefaultSettings() };

    // If setting 0 set
    if (settingsString.size() >= 1 && settingsString[0].size() >= 0)
    {
      settings.m_exponent = strToFloat(settingsString[0]);
    }

    return settings;
  }


  // Attributes
private:
  std::vector<float> _exponents;

};