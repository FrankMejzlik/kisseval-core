#pragma once

#include <string>
using namespace std::literals;
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <stack>
#include <tuple>
#include <vector>

#include "KeywordsContainer.h"
#include "common.h"
#include "config.h"
#include "data_format_config.h"
#include "utility.h"

class ImageRanker;

class FileParser
{
 public:
  static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::pair<FrameId, float>>>>
  ParseRawScoringData_ViretFormat(const std::string& inputFilepath);
  static std::vector<std::vector<float>> ParseSoftmaxBinFile_ViretFormat(const std::string& inputFilepath);
  static std::vector<std::vector<float>> ParseDeepFeasBinFile_ViretFormat(const std::string& inputFilepath);
  static std::tuple<std::string, std::map<size_t, Keyword*>, std::map<size_t, Keyword*>,
                    std::vector<std::pair<size_t, Keyword*>>, std::vector<std::unique_ptr<Keyword>>>
  ParseKeywordClassesFile_ViretFormat(const std::string& filepath);

  FileParser(ImageRanker* pRanker);

  std::tuple<VideoId, ShotId, FrameNumber> ParseVideoFilename(const std::string& filename) const;
  VideoId GetVideoIdFromFrameFilename(const std::string& filename) const;
  ShotId GetShotIdFromFrameFilename(const std::string& filename) const;

  std::vector<SelFrame> ParseImagesMetaData(const std::string& idToFilename, size_t imageIdStride = 1) const;

  // =================================
  // =================================
  // =================================
  void ProcessVideoShotsStack(std::stack<SelFrame*>& videoFrames) const;

  std::vector<ImageIdFilenameTuple> GetImageFilenames(const std::string& _imageToIdMapFilepath) const;

  bool ParseWordToVecFile(DataName data_name, std::vector<std::unique_ptr<Keyword>>& keywordsCont,
                          const std::string& filename);

  bool ParseRawScoringData_ViretFormat(DataName data_name, const std::string& inputFilepath) const;

  bool ParseSoftmaxBinFile_ViretFormat(DataName data_name, const std::string& inputFilepath) const;

  bool ParseSoftmaxBinFile_GoogleAiVisionFormat(DataName data_name, const std::string& inputFilepath) const;

  bool ParseRawScoringData_GoogleAiVisionFormat(DataName data_name, const std::string& inputFilepath) const;

  std::tuple<std::string, std::map<size_t, Keyword*>, std::map<size_t, Keyword*>,
             std::vector<std::pair<size_t, Keyword*>>, std::vector<std::unique_ptr<Keyword>>>
  ParseKeywordClassesFile_GoogleAiVisionFormat(const std::string& filepath) const;

 private:
  ImageRanker* _pRanker;
};
