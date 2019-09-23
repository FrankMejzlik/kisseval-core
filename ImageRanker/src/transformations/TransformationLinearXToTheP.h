#pragma once

#include "TransformationFunctionBase.h"

/*!
 * Transformation doing linear scale to [0, 1] and normalization
 */
class TransformationLinearXToTheP :
  public TransformationFunctionBase
{
public:
  struct Settings
  {
    Settings():
      m_exponent(1.0f),
      m_vectorIndex(0),
      m_summedHypernyms(0)
    { }
    float m_exponent;
    size_t m_vectorIndex;
    size_t m_summedHypernyms;
  };


  // Methods
public:
  TransformationLinearXToTheP() :
    TransformationFunctionBase(InputDataTransformId::cXToTheP),
    _exponents({ 1.0f})
    // \todo Implement exponent rigorously mathematically
  {}

 
  virtual bool CalculateTransformedVectors(const std::vector<std::unique_ptr<Image>>& images) const
  {
    // Itarate over all images
    for (auto&& img : images)
    {
      // Iterate over all input datasets      
      for (auto&&[kwScDataId, rawDataVector] : img->_rawImageScoringData)
      {
        // Get data info
        auto dataInfo{ img->_rawImageScoringDataInfo.at(kwScDataId) };
      
        // Calculate total sum of this bin vector
        float totalSum{ 0ULL };
        for (auto&& bin : rawDataVector)
        {
          // If normal value
          if (bin >= dataInfo.m_min)
          {
            totalSum += ((bin - dataInfo.m_min) / (dataInfo.m_max - dataInfo.m_min));
          }
          // Else zero epsilon substitution
          else 
          {
            totalSum += bin;
          }
        }

        // Iterate over all wanted exponents
        size_t i{ 0ULL };
        for (auto&& exp : _exponents)
        {
          std::vector<float> transformedDataVector;
          transformedDataVector.reserve(rawDataVector.size());

          for (auto&& bin : rawDataVector)
          {
            if (bin >= dataInfo.m_min)
            {
              transformedDataVector.emplace_back(
                ((bin - dataInfo.m_min) / (dataInfo.m_max - dataInfo.m_min)) / totalSum
              );
            }
            else {
              transformedDataVector.emplace_back(bin);
            }
          }

          // Create one copy for MAX based precalculations ->  10^1 = 1
          img->_transformedImageScoringData[kwScDataId].emplace(GetGuid(i + 10), transformedDataVector);

          // Second one for SUM based precalculations ->
          img->_transformedImageScoringData[kwScDataId].emplace(GetGuid(i), std::move(transformedDataVector));

          ++i;
        }
      }
    }

    return true;
  }

  /*!
   * 
   * 
   * \param images
   * \param settings 
   *    LSb - 0: 
   *      0 -> Precompute SUM based data vector
   *      1 -> Precompute MAX based data vector
   * \return 
   */
  virtual bool LowMem_CalculateTransformedVectors(const std::vector<std::unique_ptr<Image>>& images, size_t settings) const
  {
    LOG_ERROR("Not implemented");
    return false;
    /*
    // Itarate over all images
    for (auto&&[imgId, img] : images)
    {
      // Calculate total sum of this bin vector
      float totalSum{ 0ULL };
      for (auto&& bin : img->m_rawNetRanking)
      {
        totalSum += ((bin - img->m_min) / (img->m_max - img->m_min));
      }

      // Iterate over all wanted exponents
      size_t i{ 0ULL };
      for (auto&& exp : _exponents)
      {
        // Allow only one exponent
        if (i >= 1)
        {
          break;
        }

        {
          size_t j{ 0_z };
          for (auto&& bin : img->m_rawNetRanking)
          {
            // Do final transformation f(x) = x ^ exp
            img->m_rawNetRanking[j] = pow((((bin - img->m_min) / (img->m_max - img->m_min)) / totalSum), exp);

            ++j;
          }
        }


        // If only SUM based data vector wanted
        if (settings % 2 == 0)
        {
          // Move source vector to new destination
          img->_transformedImageScoringData.emplace(GetGuid(i), std::move(img->m_rawNetRanking));
        }
        // If only MAX based data vector wanted
        else 
        {
          // Move source vector to new destination
          img->_transformedImageScoringData.emplace(GetGuid(i + 10), std::move(img->m_rawNetRanking));
        }

        ++i;
      }
    }

    return true; 
    */
  }

  virtual size_t GetGuidFromSettings() const override
  {
    return GetGuid(_settings.m_vectorIndex + (_settings.m_summedHypernyms * 10));
  }




  Settings GetDefaultSettings() const
  {
    // Return default settings instance
    return Settings();
  }

  virtual void SetTransformationSettings(RankingModelSettings settingsString) override
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

    // If setting 1 set
    if (settingsString.size() >= 2 && settingsString[1].size() >= 0)
    {
      _settings.m_summedHypernyms = strToInt(settingsString[1]);

    }
  }



 

  // Attributes
private:
  Settings _settings;

  std::vector<float> _exponents;

};