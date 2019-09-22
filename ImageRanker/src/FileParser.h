#pragma once

#include <string>
using namespace std::literals;
#include <memory>
#include <map>
#include <vector>
#include <tuple>
#include <fstream>
#include <array>
#include <stack>

#include "config.h"
#include "common.h"
#include "utility.h"
#include "data_format_config.h"

class Image;
class ImageRanker;


class FileParser
{
public:
  FileParser(ImageRanker* pRanker);
  ~FileParser() noexcept;

  std::tuple<size_t, size_t, size_t> ParseVideoFilename(const std::string& filename) const;
  size_t GetVideoIdFromFrameFilename(const std::string& filename) const;
  size_t GetShotIdFromFrameFilename(const std::string& filename) const;

  int32_t ParseIntegerLE(const std::byte* pFirstByte) const;
  float ParseFloatLE(const std::byte* pFirstByte) const;

  std::array<char, 4> floatToBytesLE(float number)
  {
    std::array<char, 4> byteArray;

    char* bitNumber{ reinterpret_cast<char*>(&number) };

    std::get<0>(byteArray) = bitNumber[0];
    std::get<1>(byteArray) = bitNumber[1];
    std::get<2>(byteArray) = bitNumber[2];
    std::get<3>(byteArray) = bitNumber[3];

    return byteArray;
  }

  std::array<char, 4> uint32ToBytesLE(uint32_t number)
  {
    std::array<char, 4> byteArray;

    char* bitNumber{ reinterpret_cast<char*>(&number) };

    std::get<0>(byteArray) = bitNumber[0];
    std::get<1>(byteArray) = bitNumber[1];
    std::get<2>(byteArray) = bitNumber[2];
    std::get<3>(byteArray) = bitNumber[3];

    return byteArray;
  }

  void ProcessVideoShotsStack(std::stack<Image*>& videoFrames) const;

  std::vector<ImageIdFilenameTuple> GetImageFilenames(const std::string& _imageToIdMapFilepath) const;

  bool LowMem_ParseRawScoringData_ViretFormat(
    std::vector<std::unique_ptr<Image>>& imagesCont,
    KwScoringDataId kwScDataId,
    const std::string& inputFilepath
  ) const;

  std::vector<std::unique_ptr<Image>> ParseImagesMetaData(
    const std::string& idToFilename, size_t imageIdStride
  ) const;


  bool ParseRawScoringData_ViretFormat(
    std::vector<std::unique_ptr<Image>>& imagesCont,
    KwScoringDataId kwScDataId,
    const std::string& inputFilepath
  ) const;

  bool ParseSoftmaxBinFile_ViretFormat(
    std::vector<std::unique_ptr<Image>>& imagesCont,
    KwScoringDataId kwScDataId,
    const std::string& inputFilepath
  ) const;

  bool ParseSoftmaxBinFile_GoogleAiVisionFormat(
    std::vector<std::unique_ptr<Image>>& imagesCont,
    KwScoringDataId kwScDataId,
    const std::string& inputFilepath
  ) const;
  
  bool ParseRawScoringData_GoogleAiVisionFormat(
    std::vector<std::unique_ptr<Image>>& imagesCont,
    KwScoringDataId kwScDataId,
    const std::string& inputFilepath
  ) const;

  std::tuple<
    std::string,
    std::map<size_t, Keyword*>,
    std::map<size_t, Keyword*>,
    std::vector<std::pair<size_t, Keyword*>>,
    std::vector<std::unique_ptr<Keyword>>>
  ParseKeywordClassesFile_ViretFormat(const std::string& filepath) const;

  std::tuple<
    std::string,
    std::map<size_t, Keyword*>,
    std::map<size_t, Keyword*>,
    std::vector<std::pair<size_t, Keyword*>>,
    std::vector<std::unique_ptr<Keyword>>>
  ParseKeywordClassesFile_GoogleAiVisionFormat(const std::string& filepath) const;


private:
  ImageRanker* _pRanker;

};

