#pragma once

#include <string>
using namespace std::literals;
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <stack>
#include <tuple>
#include <vector>

#include "KeywordsContainer.h"
#include "common.h"
#include "config.h"
#include "utility.h"

namespace image_ranker
{
class ImageRanker;

/**
 * Purely static class providing functions for file parsing.
 */
class FileParser
{
 public:
  /**
   * Parses float matrix from a binary file that is written in row-major format and starts at `begin_offset` offset.
   *
   * FORMAT:
   *    Matrix of 4B floats, row - major:
   *    - each line is dim_N * 4B floats
   *    - number of lines is number of selected frames
   *
   * \throws std::runtime_error If error while opening/reading/closing the file.
   *
   * \param filepath Target file filepath.
   * \param row_dim   Dimension of the matrix rows.
   * \param begin_offset   Offset where actual matrix data start.
   * \return  Parsed matrix.
   */
  static Matrix<float> parse_float_matrix(const std::string& filepath, size_t row_dim, size_t begin_offset = 0);

  /**
   * Parses float vector from a binary file that starts at `begin_offset` offset.
   *
   * FORMAT:
   *    Matrix of 4B floats:
   *    - each line is dim_N * 4B floats
   *    - number of lines is number of selected frames
   *
   * \throws std::runtime_error If error while opening/reading/closing the file.
   *
   * \param filepath Target file filepath.
   * \param row_dim   Dimension of the matrix rows.
   * \param begin_offset   Offset where actual matrix data start.
   * \return  Parsed vector.
   */
  static std::vector<float> parse_float_vector(const std::string& filepath, size_t dim, size_t begin_offset = 0);

  /**
   * Parses text file mapping words to their indices (IDs).
   *
   * FORMAT: (one record per line)
   *  <word_string>   <decimal_number_ID>
   *
   * \param filepath Target file filepath.
   * \return  Parsed dictionary `word -> ID`.
   */
  static std::map<std::string, size_t> parse_w2vv_word_to_idx_file(const std::string& filepath);

  static std::pair<Matrix<float>, DataParseStats> parse_VIRET_format_frame_vector_file(const std::string& inputFilepath,
                                                                                       size_t num_frames);

  static std::pair<Matrix<float>, DataParseStats> parse_GoogleVision_format_frame_vector_file(
      const std::string& inputFilepath, size_t num_frames);

  static ViretKeywordClassesParsedData parse_VIRET_format_keyword_classes_file(const std::string& filepath,
                                                                               bool first_col_is_ID = false);

  static std::tuple<VideoId, ShotId, FrameNumber> parse_video_filename_string(const std::string& filename,
                                                                              const FrameFilenameOffsets& offsets);

  static std::vector<SelFrame> parse_image_metadata(const std::string& idToFilename,
                                                    const FrameFilenameOffsets& offsets, size_t imageIdStride = 1);

  static void process_shot_stack(std::stack<SelFrame*>& videoFrames);

  static std::vector<ImageIdFilenameTuple> get_image_filenames(const std::string& frame_to_ID_map_fpth);
};
}  // namespace image_ranker