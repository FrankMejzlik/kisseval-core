#pragma once

#include "RankingModelBase.h"

class ViretModel :
  public RankingModelBase
{
public:
  enum class eQueryOperations
  {
    cMultiplyAdd,
    cAddOnly
  };

  struct Settings
  {
    float m_trueTreshold;
    eQueryOperations m_queryOperation;
  };

public:
  Settings GetDefaultSettings() const
  {
    // Return default settings instance
    return Settings{ 0.01f , eQueryOperations::cMultiplyAdd };
  }

  Settings DecodeModelSettings(ModelSettings settingsString) const
  {
    auto settings{ GetDefaultSettings() };

    // If setting 0 set
    if (settingsString.size() >= 1 && settingsString[0].size() >= 0)
    {
      settings.m_trueTreshold = strToFloat(settingsString[0]);
    }
    // If setting 1 set
    if (settingsString.size() >= 2 && settingsString[1].size() >= 0)
    {
      settings.m_queryOperation = static_cast<eQueryOperations>(strToInt(settingsString[1]));
    }

    return settings;
  }


public:
  static float m_trueTresholdFrom;
  static float m_trueTresholdTo;
  static float m_trueTresholdStep;
  static std::vector<float> m_trueTresholds;

  static std::vector<uint8_t> m_queryOperations;
};
