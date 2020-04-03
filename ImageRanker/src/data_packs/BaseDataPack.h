#pragma once

#include <string>

#include "common.h"

class BaseDataPack
{
  [[nodiscard]] 
  virtual RankingResult rank_frames(const std::vector<std::string>& user_queries, PackModelCommands model_commands,
                                size_t result_size, FrameId target_image_ID = ERR_VAL<FrameId>()) const = 0;
};

