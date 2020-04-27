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

namespace image_ranker
{
class ImageRanker;

class FileParser
{
 public:
  /**
   * Parses float matrix from a binary file that is written in row-major format and starts at `begin_offset` offset.k
   * FORMAT:
   *    Matrix of 4B floats, row - major:
   *    - each line is dim_N * 4B floats
   *    - number of lines is number of selected frames
   */
  static std::vector<std::vector<float>> parse_float_matrix(const std::string& filepath, uint32_t row_dim,
                                                            size_t begin_offset = 0_z);

  /**
   * FORMAT:
   *    Matrix of 4B floats:
   *    - each line is dim_N * 4B floats
   *    - number of lines is number of selected frames
   */
  static std::vector<float> parse_float_vector(const std::string& filepath, uint32_t dim, uint32_t begin_offset = 0);

  static std::map<std::string, uint32_t> parse_w2vv_word_to_idx_file(const std::string& filepath);

  static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::pair<FrameId, float>>>>
  ParseRawScoringData_ViretFormat(const std::string& inputFilepath);
  static std::vector<std::vector<float>> ParseSoftmaxBinFile_ViretFormat(const std::string& inputFilepath);
  static std::vector<std::vector<float>> ParseDeepFeasBinFile_ViretFormat(const std::string& inputFilepath);
  static Matrix<float> ParseRawScoringData_GoogleAiVisionFormat(const std::string& inputFilepath);
  static std::tuple<std::string, std::map<size_t, Keyword*>, std::map<size_t, Keyword*>,
                    std::vector<std::pair<size_t, Keyword*>>, std::vector<std::unique_ptr<Keyword>>,
                    std::map<KeywordId, Keyword*>>
  ParseKeywordClassesFile_ViretFormat(const std::string& filepath, bool first_col_is_ID = false);

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
}  // namespace image_ranker