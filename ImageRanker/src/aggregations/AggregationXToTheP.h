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
    size_t m_vectorIndex;
  };


  // Methods
public:
  AggregationXToTheP() :
    AggregationFunctionBase(AggregationId::cXToTheP),
    _exponents({ 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f})
    //_exponents({ 0.2f, 1.0f, 2.0f })
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
        totalSum += (bin - img->m_min);
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
          aggVector.emplace_back( pow((bin - img->m_min)/ totalSum, exp));
        }

        // Insert this aggregation to IR agregations
        img->m_aggVectors.emplace(GetGuid(i), std::move(aggVector));
        
        ++i;
      } 
    }

    return true;
  }

  virtual size_t GetGuidFromSettings() const override
  {
    return GetGuid(_settings.m_vectorIndex);
  }




  Settings GetDefaultSettings() const
  {
    // Return default settings instance
    return Settings{ 1.0f };
  }

  virtual void SetAggregationSettings(ModelSettings settingsString) override
  {
    // If setting 0 set
    if (settingsString.size() >= 1 && settingsString[0].size() >= 0)
    {
      _settings.m_exponent = strToFloat(settingsString[0]);

      // Find out what index it is
      size_t i{0ULL};
      bool found{ false };
      for (auto&& exp : _exponents)
      {
        // If this exp
        if ((_settings.m_exponent - exp) < 0.000001f)
        {
          found = true;
          break;
        }

        ++i;
      }

      // If this exponent found
      if (found) 
      {
        _settings.m_vectorIndex = i;
      }
      else 
      {
        _settings.m_vectorIndex = SIZE_T_ERROR_VALUE;
        LOG_ERROR("Unknown exponent in XToTheP aggregation.");
      }

    }
  }



 

  // Attributes
private:
  Settings _settings;

  std::vector<float> _exponents;

};