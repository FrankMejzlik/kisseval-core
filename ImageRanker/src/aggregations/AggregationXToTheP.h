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
    //_exponents({ 0.2f, 0.5f, 0.8f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 9.0f }),
    _exponents({ 0.2f, 1.0f, 2.0f })
  {
  }

  enum eExponents 
  {
    
  };
 
  virtual bool CalculateTransformedVectors(const std::unordered_map<size_t, std::unique_ptr<Image>>& images) const
  {
    // Itarate over all images
    for (auto&& [imgId, img] : images)
    {
      // Calculate total sum of this bin vector
      float totalSum{ 0ULL };
      for (auto&& bin : img->m_rawNetRanking)
      {
        totalSum += bin;
      }

      // Iterate over all wanted exponents
      size_t i{0ULL};
      for (auto&& exp : _exponents)
      {
        std::vector<float> aggVector;
        aggVector.reserve(img->GetNumBins());

        for (auto&& bin : img->m_rawNetRanking)
        {
          // Do final transformation f(x) = x ^ exp
          // xoxo: power after of before dividing with total sum?
          aggVector.emplace_back( pow(bin / totalSum, exp));
        }

        // Insert this aggregation to IR agregations
        img->m_aggVectors.emplace(GetGuid(i), std::move(aggVector));
        
        ++i;
      } 
    }

    return true;
  }



  Settings GetDefaultSettings() const
  {
    // Return default settings instance
    return Settings{ 1.0f };
  }

  virtual void SetAggregationSettings(ModelSettings settingsString) override
  {
    auto settings{ GetDefaultSettings() };

    // If setting 0 set
    if (settingsString.size() >= 1 && settingsString[0].size() >= 0)
    {
      settings.m_exponent = strToFloat(settingsString[0]);
    }
  }



 

  // Attributes
private:
  Settings _settings;

  std::vector<float> _exponents;

};