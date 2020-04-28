

#include "json.hpp"
using json = nlohmann::json;

#include "data_packs/Google_based/GoogleVisionDataPack.h"
#include "data_packs/VIRET_based/ViretDataPack.h"
#include "data_packs/W2VV_based/W2vvDataPack.h"

#include "ImageRanker.h"

#include "datasets/SelFramesDataset.h"

using namespace image_ranker;

ImageRanker::Config ImageRanker::parse_data_config_file(eMode mode, const std::string& filepath,
                                                        const std::string& data_dir)
{
  // Read the JSON cfg file
  std::ifstream i(filepath);
  json json_data;
  i >> json_data;

  // Imagesets
  std::vector<DatasetDataPackRef> imagesets;
  for (auto&& is1 : json_data["imagesets"])
  {
    // Skip inactive
    if (!is1["active"].get<bool>() || is1.is_null())
    {
      continue;
    }

    imagesets.emplace_back(DatasetDataPackRef{
        is1["ID"].get<std::string>(), is1["description"].get<std::string>(), is1["ID"].get<std::string>(),
        data_dir + is1["frames_dir"].get<std::string>(), data_dir + is1["ID_to_frame_fpth"].get<std::string>()});
  }

  // VIRET data packs
  std::vector<ViretDataPackRef> VIRET_data_packs;
  for (auto&& dp1 : json_data["data_packs"]["VIRET_based"])
  {
    // Skip inactive
    if (!dp1["active"].get<bool>() || dp1.is_null())
    {
      continue;
    }

    VIRET_data_packs.emplace_back(ViretDataPackRef{
        dp1["ID"].get<std::string>(), dp1["description"].get<std::string>(), dp1["model_options"].get<std::string>(),
        dp1["data"]["target_dataset"].get<std::string>(),

        dp1["vocabulary"]["ID"].get<std::string>(), dp1["vocabulary"]["description"].get<std::string>(),
        data_dir + dp1["vocabulary"]["keyword_synsets_fpth"].get<std::string>(),

        data_dir + dp1["data"]["presoftmax_scorings_fpth"].get<std::string>(),
        data_dir + dp1["data"]["softmax_scorings_fpth"].get<std::string>(),
        data_dir + dp1["data"]["deep_features_fpth"].get<std::string>()});
  }

  // Google data packs
  std::vector<GoogleDataPackRef> Google_data_packs;
  for (auto&& dp3 : json_data["data_packs"]["Google_based"])
  {
    // Skip inactive
    if (!dp3["active"].get<bool>() || dp3.is_null())
    {
      continue;
    }

    Google_data_packs.emplace_back(GoogleDataPackRef{
        dp3["ID"].get<std::string>(), dp3["description"].get<std::string>(), dp3["model_options"].get<std::string>(),
        dp3["data"]["target_dataset"].get<std::string>(),

        dp3["vocabulary"]["ID"].get<std::string>(), dp3["vocabulary"]["description"].get<std::string>(),
        data_dir + dp3["vocabulary"]["keyword_synsets_fpth"].get<std::string>(),

        data_dir + dp3["data"]["presoftmax_scorings_fpth"].get<std::string>()});
  }

  // W2VV data packs
  std::vector<W2vvDataPackRef> W2VV_data_packs;
  for (auto&& dp3 : json_data["data_packs"]["W2VV_based"])
  {
    // Skip inactive
    if (!dp3["active"].get<bool>() || dp3.is_null())
    {
      continue;
    }

    W2VV_data_packs.emplace_back(
        W2vvDataPackRef{dp3["ID"].get<std::string>(),
                        dp3["description"].get<std::string>(),
                        dp3["model_options"].get<std::string>(),
                        dp3["data"]["target_dataset"].get<std::string>(),

                        dp3["vocabulary"]["ID"].get<std::string>(),
                        dp3["vocabulary"]["description"].get<std::string>(),

                        data_dir + dp3["vocabulary"]["keyword_synsets_fpth"].get<std::string>(),

                        data_dir + dp3["vocabulary"]["keyword_features_fpth"].get<std::string>(),
                        dp3["vocabulary"]["keyword_features_dim"].get<size_t>(),
                        dp3["vocabulary"]["keyword_features_data_offset"].get<size_t>(),

                        data_dir + dp3["vocabulary"]["keyword_bias_vec_fpth"].get<std::string>(),
                        dp3["vocabulary"]["keyword_bias_vec_dim"].get<size_t>(),
                        dp3["vocabulary"]["keyword_bias_vec_data_offset"].get<size_t>(),

                        data_dir + dp3["vocabulary"]["keyword_PCA_mat_fpth"].get<std::string>(),
                        dp3["vocabulary"]["keyword_PCA_mat_dim"].get<size_t>(),
                        dp3["vocabulary"]["keyword_PCA_mat_data_offset"].get<size_t>(),

                        data_dir + dp3["vocabulary"]["keyword_PCA_mean_vec_fpth"].get<std::string>(),
                        dp3["vocabulary"]["keyword_PCA_mean_vec_dim"].get<size_t>(),
                        dp3["vocabulary"]["keyword_PCA_mean_vec_offset"].get<size_t>(),

                        data_dir + dp3["data"]["deep_features_fpth"].get<std::string>(),
                        dp3["data"]["deep_features_dim"].get<size_t>(),
                        dp3["data"]["deep_features_data_offset"].get<size_t>()});
  }

  return {ImageRanker::eMode::cFullAnalytical, imagesets, VIRET_data_packs, Google_data_packs, W2VV_data_packs};
}
ImageRanker::ImageRanker(const ImageRanker::Config& cfg) : _settings(cfg), _fileParser(this), _data_manager(this)
{
  /*
   * Load all available datasets
   */
  for (auto&& pack : _settings.config.dataset_packs)
  {
    // Initialize all images
    auto frames = _fileParser.ParseImagesMetaData(pack.imgage_to_ID_fpth, 1);

    _imagesets.emplace(pack.ID, std::make_unique<SelFramesDataset>(pack.ID, pack.images_dir, std::move(frames)));
  }

  /*
   * Load all available data packs
   */
  // VIRET type
  for (auto&& pack : _settings.config.VIRET_packs)
  {
    // Initialize all images
    auto presoft_data = FileParser::ParseSoftmaxBinFile_ViretFormat(pack.score_data.presoftmax_scorings_fpth);
    auto soft_data = FileParser::ParseSoftmaxBinFile_ViretFormat(pack.score_data.softmax_scorings_fpth);
    auto deep_features = FileParser::ParseDeepFeasBinFile_ViretFormat(pack.score_data.deep_features_fpth);

    const auto& is{imageset(pack.target_imageset)};

    _data_packs.emplace(pack.ID,
                        std::make_unique<ViretDataPack>(is, pack.ID, pack.target_imageset, pack.model_options,
                                                        pack.description, pack.vocabulary_data, std::move(presoft_data),
                                                        std::move(soft_data), std::move(deep_features)));
  }

  // Google type
  for (auto&& pack : _settings.config.Google_packs)
  {
    const auto& is{imageset(pack.target_imageset)};

    // Initialize all images
    // \todo Use transparent sparse matrix representation for Google data
    auto presoft_data = FileParser::ParseRawScoringData_GoogleAiVisionFormat(pack.score_data.presoftmax_scorings_fpth);

    _data_packs.emplace(pack.ID, std::make_unique<GoogleVisionDataPack>(is, pack.ID, pack.target_imageset,
                                                                        pack.model_options, pack.description,
                                                                        pack.vocabulary_data, std::move(presoft_data)));
  }

  // W2VV type
  for (auto&& pack : _settings.config.W2VV_packs)
  {
    const auto& is{imageset(pack.target_imageset)};

    // Keyword feature vectors
    auto kw_features =
        FileParser::parse_float_matrix(pack.vocabulary_data.kw_features_fpth, pack.vocabulary_data.kw_features_dim,
                                       pack.vocabulary_data.kw_features_data_offset);
    // Keyword bias vector
    auto bias_vec_transposed =
        FileParser::parse_float_vector(pack.vocabulary_data.kw_bias_vec_fpth, pack.vocabulary_data.kw_bias_vec_dim,
                                       pack.vocabulary_data.kw_bias_vec_data_offset);

    // Keyword PCA matrix (2048 -> 128)
    auto PCA_mat =
        FileParser::parse_float_matrix(pack.vocabulary_data.kw_PCA_mat_fpth, pack.vocabulary_data.kw_PCA_mat_dim,
                                       pack.vocabulary_data.kw_PCA_mat_data_offset);
    auto PCA_mean_vec = FileParser::parse_float_vector(pack.vocabulary_data.kw_PCA_mean_vec_fpth,
                                                       pack.vocabulary_data.kw_PCA_mean_vec_dim,
                                                       pack.vocabulary_data.kw_PCA_mean_vec_data_offset);

    // Frame feature vectors
    auto deep_features = FileParser::parse_float_matrix(
        pack.score_data.img_features_fpth, pack.score_data.img_features_dim, pack.score_data.img_features_offset);
    // Normalize rows
    deep_features = normalize_matrix_rows(std::move(deep_features));

    _data_packs.emplace(pack.ID, std::make_unique<W2vvDataPack>(
                                     is, pack.ID, pack.target_imageset, pack.model_options, pack.description,
                                     pack.vocabulary_data, std::move(deep_features), std::move(kw_features),
                                     std::move(bias_vec_transposed), std::move(PCA_mat), std::move(PCA_mean_vec)));
  }
}

std::vector<GameSessionQueryResult> ImageRanker::submit_annotator_user_queries(
    const StringId& data_pack_ID, const ::std::string& model_options, size_t user_level, bool with_example_images,
    const std::vector<AnnotatorUserQuery>& user_queries)
{
  auto res = _data_packs.find(data_pack_ID);
  if (res == _data_packs.end())
  {
    LOGW("Accessing non-existent data pack '" << data_pack_ID << "'");
    return std::vector<GameSessionQueryResult>();
  }

  _data_manager.submit_annotator_user_queries(data_pack_ID, res->second->get_vocab_ID(), model_options, user_level,
                                              with_example_images, user_queries);

  /*
   * Construct result for the user
   */
  std::vector<GameSessionQueryResult> userResult;
  userResult.reserve(user_queries.size());

  const auto& dp = data_pack(data_pack_ID);
  const auto& is = imageset(dp.target_imageset_ID());

  for (auto&& query : user_queries)
  {
    GameSessionQueryResult result;

    result.session_ID = query.session_ID;
    result.human_readable_query = query.user_query_readable.at(0);
    result.frame_filename = ((*is)[query.target_sequence_IDs.at(0)]).m_filename;

    auto top_KWs = dp.top_frame_keywords(query.target_sequence_IDs.at(0));

    std::stringstream model_top_query_ss;

    for (auto&& KW : top_KWs)
    {
      model_top_query_ss << KW->m_word << ", ";
    }

    result.model_top_query = model_top_query_ss.str();

    userResult.emplace_back(std::move(result));
  }

  return userResult;
}

const std::string& ImageRanker::get_frame_filename(const std::string& imageset_ID, size_t imageId) const
{
  const SelFrame& img = get_frame(imageset_ID, imageId);

  return img.m_filename;
}

const SelFrame& ImageRanker::get_frame(const std::string& imageset_ID, size_t imageId) const
{
  return (*imageset(imageset_ID))[imageId];
}

std::vector<const SelFrame*> ImageRanker::get_random_frame_sequence(const std::string& imageset_ID,
                                                                    size_t seq_len) const
{
  std::vector<const SelFrame*> result_frame_ptrs;

  const SelFrame* p_first_frame = nullptr;

  // Get the first frame with enough successors
  do
  {
    p_first_frame = get_random_frame(imageset_ID);
  } while (p_first_frame->m_num_successors < seq_len);

  result_frame_ptrs.emplace_back(p_first_frame);

  // Get his successors
  for (size_t i = 1; i < seq_len; ++i)
  {
    const SelFrame& frame = get_frame(imageset_ID, p_first_frame->m_ID + i);
    result_frame_ptrs.emplace_back(&frame);
  }

  return result_frame_ptrs;
}

const SelFrame* ImageRanker::get_random_frame(const std::string& imageset_ID) const
{
  return &(imageset(imageset_ID)->random_frame());
}

AutocompleteInputResult ImageRanker::get_autocomplete_results(const std::string& data_pack_ID,
                                                              const std::string& query_prefix, size_t result_size,
                                                              bool with_example_image) const
{
  // Force lowercase
  std::locale loc;
  std::string lower;

  // Convert to lowercase
  for (auto elem : query_prefix)
  {
    lower.push_back(std::tolower(elem, loc));
  }

  const BaseDataPack& dp = data_pack(data_pack_ID);

  return dp.get_autocomplete_results(query_prefix, result_size, with_example_image);
}

LoadedImagesetsInfo ImageRanker::get_loaded_imagesets_info() const
{
  std::vector<ImagesetInfo> infos;

  for (auto&& is : _imagesets)
  {
    infos.emplace_back(is.second->get_info());
  }

  return LoadedImagesetsInfo{infos};
}

LoadedDataPacksInfo ImageRanker::get_loaded_data_packs_info() const
{
  std::vector<DataPackInfo> infos;

  for (auto&& dp : _data_packs)
  {
    infos.emplace_back(dp.second->get_info());
  }

  return LoadedDataPacksInfo{infos};
}

RankingResult ImageRanker::rank_frames(const std::vector<std::string>& user_queries, const DataPackId& data_pack_ID,
                                       const PackModelCommands& model_commands, size_t result_size,
                                       bool native_lang_queries, FrameId target_image_ID) const
{
  const BaseDataPack& dp{data_pack(data_pack_ID)};

  // Decide if native or ID based version wanted
  if (native_lang_queries)
  {
    return dp.rank_frames(user_queries, model_commands, result_size, target_image_ID);
  }

  // Parse CNF strings
  std::vector<CnfFormula> cnf_user_query;
  cnf_user_query.reserve(user_queries.size());
  for (auto&& single_query : user_queries)
  {
    cnf_user_query.emplace_back(parse_cnf_string(single_query));
  }

  return dp.rank_frames(cnf_user_query, model_commands, result_size, target_image_ID);
}

ModelTestResult ImageRanker::run_model_test(eUserQueryOrigin queries_origin, const DataPackId& data_pack_ID,
                                            const PackModelCommands& model_commands, bool native_lang_queries,
                                            size_t num_points, bool normalize_y) const
{
  const auto& dp{data_pack(data_pack_ID)};
  const auto& is{imageset(dp.target_imageset_ID())};

  ModelTestResult res;
  size_t num_queries{ERR_VAL<size_t>()};

  // Decide if native or ID based version wanted
  if (native_lang_queries)
  {
    // Fetch queries from the DB
    auto native_test_queries{_data_manager.fetch_user_native_test_queries(queries_origin)};
    num_queries = native_test_queries.size();

    res = dp.test_model(native_test_queries, model_commands, num_points);
  }
  else
  {
    // Fetch queries from the DB
    auto test_queries{_data_manager.fetch_user_test_queries(queries_origin, dp.get_vocab_ID())};
    num_queries = test_queries.size();

    res = dp.test_model(test_queries, model_commands, num_points);
  }

  if (normalize_y)
  {
    std::transform(res.begin(), res.end(), res.begin(), [num_queries](const std::pair<uint32_t, uint32_t>& x) {
      return std::pair(x.first, uint32_t((float(x.second) / num_queries) * 100.0F));
    });
  }

  return res;
}

// =====================================
//  NOT REFACTORED CODE BELOW
// =====================================

#if 0

std::tuple<KeywordsGeneralStatsTuple, ScoringsGeneralStatsTuple, AnnotatorDataGeneralStatsTuple,
           RankerDataGeneralStatsTuple>
ImageRanker::GetGeneralStatistics(DataId data_ID, UserDataSourceId dataSourceType) const
{
  // Goes first
  AnnotatorDataGeneralStatsTuple annotatorStatsTuple{GetGeneralAnnotatorDataStatistics(data_ID, dataSourceType)};

  KeywordsGeneralStatsTuple keywordsStatsTuple{GetGeneralKeywordsStatistics(data_ID, dataSourceType)};
  ScoringsGeneralStatsTuple scoringStatsTuple{GetGeneralScoringStatistics(data_ID, dataSourceType)};
  RankerDataGeneralStatsTuple rankerStatsTuple{GetGeneralRankerDataStatistics(data_ID, dataSourceType)};

  return std::tuple(std::move(keywordsStatsTuple), std::move(scoringStatsTuple), std::move(annotatorStatsTuple),
                    std::move(rankerStatsTuple));
}

KeywordsGeneralStatsTuple ImageRanker::GetGeneralKeywordsStatistics(DataId data_ID,
                                                                    UserDataSourceId dataSourceType) const
{
  KeywordsGeneralStatsTuple resultTuple;

  if (std::get<0>(data_ID) == eVocabularyId::VIRET_1200_WORDNET_2019)
  {
    std::get<0>(resultTuple) = 2008;
  }
  else
  {
    // Get number of distincts keywords
    std::get<0>(resultTuple) = GetCorrectKwContainerPtr(data_ID)->_keywords.size();
  }

  return resultTuple;
}
ScoringsGeneralStatsTuple ImageRanker::GetGeneralScoringStatistics(DataId data_ID,
                                                                   UserDataSourceId dataSourceType) const
{
  ScoringsGeneralStatsTuple resultTuple;

  return resultTuple;
}
AnnotatorDataGeneralStatsTuple ImageRanker::GetGeneralAnnotatorDataStatistics(DataId data_ID,
                                                                              UserDataSourceId dataSourceType) const
{
  AnnotatorDataGeneralStatsTuple resultTuple;

  // If not computed, do so
  if (_stat_labelHit.count(data_ID) <= 0_z)
  {
    ExportUserAnnotatorNumHits(data_ID, UserDataSourceId::cAll, "./a.tmp");
  }

  std::get<0>(resultTuple) = _stat_minLabels.at(data_ID);
  std::get<1>(resultTuple) = _stat_maxLabels.at(data_ID);
  std::get<2>(resultTuple) = _stat_avgLabels.at(data_ID);
  std::get<3>(resultTuple) = _stat_medianLabels.at(data_ID);
  std::get<4>(resultTuple) = _stat_labelHit.at(data_ID);

  return resultTuple;
}
RankerDataGeneralStatsTuple ImageRanker::GetGeneralRankerDataStatistics(DataId data_ID,
                                                                        UserDataSourceId dataSourceType) const
{
  RankerDataGeneralStatsTuple resultTuple;

  return resultTuple;
}

std::string ImageRanker::ExportDataFile(DataId data_ID, eExportFileTypeId fileType,
                                        const std::string& outputFilepath, bool native) const
{
  bool succ{true};

  try
  {
    switch (fileType)
    {
      case eExportFileTypeId::cUserAnnotatorQueries:
        succ = ExportUserAnnotatorData(data_ID, UserDataSourceId::cAll, outputFilepath, native);
        break;

      case eExportFileTypeId::cNetNormalizedScores:
        succ = ExportNormalizedScores(data_ID, outputFilepath);
        break;

      case eExportFileTypeId::cQueryNumHits:
        succ = ExportUserAnnotatorNumHits(data_ID, UserDataSourceId::cAll, outputFilepath);
        break;

      default:
        LOG_ERROR("Unknown export data type! (ImageRanker::ExportDataFile())");
    }
  }
  catch (const UnableToCreateFileExcept&)
  {
    succ = false;
  }

  if (!succ)
  {
    return ""s;
  }

  return outputFilepath;
}

bool ImageRanker::ExportUserAnnotatorData(DataId data_ID, UserDataSourceId dataSource,
                                          const std::string& outputFilepath, bool native) const
{
  std::string strType = "";
  if (dataSource == UserDataSourceId::cAll)
  {
    strType = "( type=0 OR type=1 OR type=10 OR type=11 )";
  }
  else if (dataSource == UserDataSourceId::cDeveloper)
  {
    strType = "( type=0 OR type=10 )";
  }
  else
  {
    strType = "( type=1 OR type=11 )";
  }

  // Fetch pairs of <Q, Img>
  std::string query(
      "\
        SELECT image_id, type, query  FROM `" +
      _db.GetDbName() +
      "`.queries \
          WHERE " +
      strType +
      " AND \
            keyword_data_type = " +
      std::to_string((int)std::get<0>(data_ID)) +
      " AND \
            scoring_data_type = " +
      std::to_string((int)std::get<1>(data_ID)) + ";");

  auto dbData = _db.ResultQuery(query);

  if (dbData.first != 0)
  {
    LOG_ERROR("Error getting queries from database."s);
  }

  std::ofstream outFileStream(outputFilepath);
  if (!outFileStream.is_open())
  {
    LOG_WARN("Unable to create file: " + outputFilepath);
    throw UnableToCreateFileExcept("Unable to create file: "s + outputFilepath);
  }

  for (auto&& idQueryRow : dbData.second)
  {
    // Image ID
    outFileStream << std::to_string(static_cast<size_t>(strToInt(idQueryRow[0].data())) * TEST_QUERIES_ID_MULTIPLIER)
                  << ",";

    // Source type
    outFileStream << idQueryRow[1];

    auto ids{GetCorrectKwContainerPtr(data_ID)->GetCanonicalQueryNoRecur(idQueryRow[2])};

    if (native)
    {
      size_t cnt = ids.size();
      size_t i = 0;
      outFileStream << ",\"";
      for (auto&& id : ids)
      {
        std::string str_kw = GetKeywordByWordnetId(data_ID, id);
        outFileStream << str_kw;
        if (i < cnt - 1)
        {
          outFileStream << " ";
        }
        ++i;
      }
      outFileStream << "\"";
    }
    else
    {
      for (auto&& id : ids)
      {
        outFileStream << "," << std::to_string(id);
      }
    }

    outFileStream << std::endl;
  }

  outFileStream.close();

  return true;
}

bool ImageRanker::ExportNormalizedScores(DataId data_ID, const std::string& outputFilepath) const
{
  std::ofstream outFileStream(outputFilepath);
  if (!outFileStream.is_open())
  {
    LOG_WARN("Unable to create file: " + outputFilepath);
    throw UnableToCreateFileExcept("Unable to create file: "s + outputFilepath);
  }

  bool isGoogle{std::get<0>(data_ID) == eVocabularyId::GOOGLE_AI_20K_2019 ? true : false};

  for (auto&& img : _images)
  {
    outFileStream << img->m_imageId;

    auto rawVec{img->_rawImageScoringData.at(data_ID)};

    // Normalize vector
    float totalSum{0.0f};
    for (auto&& score : rawVec)
    {
      // If google ignore zero values
      if (isGoogle && score <= GOOGLE_AI_NO_LABEL_SCORE)
      {
        continue;
      }

      totalSum += score;
    }

    size_t i{0_z};
    for (auto&& score : rawVec)
    {
      // If google ignore zero values
      if (isGoogle && score <= GOOGLE_AI_NO_LABEL_SCORE)
      {
        ++i;
        continue;
      }

      float scoreNorm{score / totalSum};

      auto kwPtr{GetCorrectKwContainerPtr(data_ID)->GetKeywordPtrByVectorIndex(i)};

      outFileStream << "," << kwPtr->m_wordnetId << "," << scoreNorm;

      ++i;
    }
    outFileStream << std::endl;
  }

  outFileStream.close();

  return true;
}

bool ImageRanker::ExportUserAnnotatorNumHits(DataId data_ID, UserDataSourceId dataSource,
                                             const std::string& outputFilepath) const
{
  std::string strType = "";
  if (dataSource == UserDataSourceId::cAll)
  {
    strType = "( type=0 OR type=1 OR type=10 OR type=11 )";
  }
  else if (dataSource == UserDataSourceId::cDeveloper)
  {
    strType = "( type=0 OR type=10 )";
  }
  else
  {
    strType = "( type=1 OR type=11 )";
  }

  bool isGoogle{std::get<0>(data_ID) == eVocabularyId::GOOGLE_AI_20K_2019 ? true : false};

  // Fetch pairs of <Q, Img>
  std::string query(
      "\
        SELECT id, image_id, type, query  FROM `" +
      _db.GetDbName() +
      "`.queries \
          WHERE " +
      strType +
      " AND \
            keyword_data_type = " +
      std::to_string((int)std::get<0>(data_ID)) +
      " AND \
            scoring_data_type = " +
      std::to_string((int)std::get<1>(data_ID)) + ";");

  auto dbData = _db.ResultQuery(query);

  if (dbData.first != 0)
  {
    LOG_ERROR("Error getting queries from database."s);
  }

  std::ofstream outFileStream(outputFilepath);
  if (!outFileStream.is_open())
  {
    LOG_WARN("Unable to create file: " + outputFilepath);
    throw UnableToCreateFileExcept("Unable to create file: "s + outputFilepath);
  }

  size_t totalTotalLabels{0_z};
  size_t totalTotalHits{0_z};
  size_t minLabels{0_z};
  size_t maxLabels{0_z};
  std::vector<size_t> labelNums;

  for (auto&& idQueryRow : dbData.second)
  {
    size_t numNetLabels{0_z};

    // Query ID
    outFileStream << idQueryRow[0].data() << ",";

    size_t imageId{static_cast<size_t>(strToInt(idQueryRow[1].data())) * TEST_QUERIES_ID_MULTIPLIER};

    // Image ID
    outFileStream << std::to_string(imageId) << ",";

    // Source type
    outFileStream << idQueryRow[2] << ",";

    auto ids{GetCorrectKwContainerPtr(data_ID)->GetCanonicalQueryNoRecur(idQueryRow[3])};

    size_t hitCount{0_z};

    auto imgPtr{GetImageDataById(imageId)};
    const std::vector<float>* scoreVector{nullptr};

    if (std::get<0>(data_ID) == eVocabularyId::VIRET_1200_WORDNET_2019)
    {
      scoreVector = &imgPtr->_transformedImageScoringData.at(data_ID).at(200);
    }
    else
    {
      scoreVector = &imgPtr->_rawImageScoringData.at(data_ID);
    }

    // Count number of label given by net
    for (auto&& score : *scoreVector)
    {
      //  If hit
      if (isGoogle)
      {
        if (score > GOOGLE_AI_NO_LABEL_SCORE)
        {
          ++numNetLabels;
        }
      }
      else
      {
        ++numNetLabels;
      }
    }

    for (auto&& id : ids)
    {
      auto ptrKw{GetCorrectKwContainerPtr(data_ID)->GetKeywordConstPtrByWordnetId(id)};

      auto score{(*scoreVector)[ptrKw->m_vectorIndex]};

      //  If hit
      if (isGoogle)
      {
        if (score > GOOGLE_AI_NO_LABEL_SCORE)
        {
          ++hitCount;
        }
      }
      else
      {
        if (score > VIRET_TRESHOLD_LINEAR_01)
        {
          ++hitCount;
        }
      }
    }

    outFileStream << hitCount << "," << ids.size() << "," << numNetLabels << std::endl;

    minLabels = std::min(minLabels, ids.size());
    maxLabels = std::max(maxLabels, ids.size());

    labelNums.emplace_back(ids.size());

    totalTotalLabels += ids.size();
    totalTotalHits += hitCount;
  }

  float avgLabels{static_cast<float>(totalTotalLabels) / dbData.second.size()};

  size_t idxFirst{dbData.second.size() / 2};
  bool numQueriesOdd{dbData.second.size() % 2 == 1 ? true : false};
  float medianLabels{0.0f};

  if (idxFirst == 0)
  {
    LOG_ERROR(
        "WROOOONG! Will go out of bounds! "
        "(ImageRanker::ExportUserAnnotatorNumHits())");
  }

  if (numQueriesOdd)
  {
    medianLabels = static_cast<float>(labelNums[idxFirst]);
  }
  else
  {
    medianLabels = static_cast<float>((labelNums[idxFirst - 1] + labelNums[idxFirst]) / 2);
  }

  // Save prob hit
  _stat_minLabels.erase(data_ID);
  _stat_minLabels.emplace(data_ID, minLabels);

  _stat_maxLabels.erase(data_ID);
  _stat_maxLabels.emplace(data_ID, maxLabels);

  _stat_avgLabels.erase(data_ID);
  _stat_avgLabels.emplace(data_ID, avgLabels);

  _stat_medianLabels.erase(data_ID);
  _stat_medianLabels.emplace(data_ID, medianLabels);

  _stat_labelHit.erase(data_ID);
  _stat_labelHit.emplace(data_ID, (float)totalTotalHits / totalTotalLabels);

  return true;
}

bool ImageRanker::InitializeFullMode()
{
  //
  // Setup supported transformations and models
  //
  {
    // Insert all desired transformations
    _transformations.emplace(InputDataTransformId::cNoTransform, std::make_unique<TransformationNoTransform>());
    _transformations.emplace(InputDataTransformId::cSoftmax, std::make_unique<TransformationSoftmax>());
    _transformations.emplace(InputDataTransformId::cXToTheP, std::make_unique<TransformationLinear>());

    // Insert all desired ranking models
    _models.emplace(RankingModelId::cViretBase, std::make_unique<MultSumMaxModel>());
    _models.emplace(RankingModelId::cBooleanBucket, std::make_unique<BooleanBucketModel>());
  }



  // Fill in scoring data to images
  for (auto&& [data_ID, filepath] : _imageScoringFileRefs)
  {
    // Choose correct parsing method
    switch (std::get<0>(data_ID))
    {
      case eVocabularyId::VIRET_1200_WORDNET_2019:
        _fileParser.ParseRawScoringData_ViretFormat(_images, data_ID, filepath);
        break;

      case eVocabularyId::GOOGLE_AI_20K_2019:
        _fileParser.ParseRawScoringData_GoogleAiVisionFormat(_images, data_ID, filepath);
        break;

      default:
        LOG_ERROR("Invalid keyword data type.");
    }
  }

  // Fill in Softmax data
  for (auto&& [data_ID, filepath] : _imageSoftmaxScoringFileRefs)
  {
    // Choose correct parsing method
    switch (std::get<0>(data_ID))
    {
      case eVocabularyId::VIRET_1200_WORDNET_2019:
        _fileParser.ParseSoftmaxBinFile_ViretFormat(_images, data_ID, filepath);
        break;

      case eVocabularyId::GOOGLE_AI_20K_2019:
        _fileParser.ParseSoftmaxBinFile_GoogleAiVisionFormat(_images, data_ID, filepath);
        break;

      default:
        LOG_ERROR("Invalid scoring data type.");
    }
  }

  // Load and process all supported transformations
  for (auto&& transFn : _transformations)
  {
    transFn.second->CalculateTransformedVectors(_images);
  }

  // Apply hypernym recalculation on all transformed vectors
  for (auto&& pImg : _images)
  {
    for (auto&& [ksScDataId, transformedData] : pImg->_transformedImageScoringData)
    {
      // \todo implement propperly

      // Copy untouched vector for simulating user from [0, 1] linear scale
      // transformation
      pImg->_rawSimUserData.emplace(ksScDataId, pImg->GetScoringVectorsConstPtr(ksScDataId)->at(200));

      if (std::get<0>(ksScDataId) == eVocabularyId::GOOGLE_AI_20K_2019)
      {
        continue;
      }

      for (auto&& [transformId, binVec] : transformedData)
      {
        // If is summ based aggregation precalculation
        if (((transformId / 10) % 10) == 0)
        {
          RecalculateHypernymsInVectorUsingSum(binVec);
        }
        else
        {
          RecalculateHypernymsInVectorUsingMax(binVec);
        }
      }
    }
  }

  //    auto result = GetImageDataById(11711);

  //// Comparator lambda for priority queue
  // auto cmp = [](const std::pair<float, size_t>& left, const std::pair<float,
  // size_t>& right)
  //{
  //  return left.first < right.first;
  //};

  //// Reserve enough space in container
  // std::vector<std::pair<float, size_t>> container;

  // std::priority_queue<
  //  std::pair<float, size_t>,
  //  std::vector<std::pair<float, size_t>>,
  //  decltype(cmp)> maxHeap(cmp, std::move(container));

  // size_t i = 0;
  // float sum = 0.0f;

  // auto vec =
  // result->GetScoringVectorsPtr(std::tuple(eKeywordsDataType::cViret1,
  // eImageScoringDataType::cNasNet));

  // for (auto&& scor : vec->at(200))
  //{
  //  maxHeap.emplace(scor, i);
  //  sum += scor;
  //  ++i;
  //}

  // for (size_t i = 0; i < 10; ++i)
  //{
  //  auto [sc, idx] {maxHeap.top()};
  //  maxHeap.pop();

  //  auto word = GetKeywordByVectorIndex(std::tuple(eKeywordsDataType::cViret1,
  //  eImageScoringDataType::cNasNet), idx);

  //  std::cout << word->m_word << "=> " << sc << std::endl;
  //
  //}
  // std::cout << sum << std::endl;

  // Calculate approx document frequency
  // ComputeApproxDocFrequency(200, TRUE_TRESHOLD_FOR_KW_FREQUENCY);

  // Initialize gridtests
  InitializeGridTests();

  LOG("Aplication initialized in FULL MODE.");
  return true;
}

TransformationFunctionBase* ImageRanker::GetAggregationById(InputDataTransformId id) const
{
  // Try to get this aggregation
  if (auto result{_transformations.find(static_cast<InputDataTransformId>(id))}; result != _transformations.end())
  {
    return result->second.get();
  }
  else
  {
    LOG_ERROR("Aggregation not found!");
    return nullptr;
  }
}

RankingModelBase* ImageRanker::GetRankingModelById(RankingModelId id) const
{
  // Try to get this aggregation
  if (auto result{_models.find(static_cast<RankingModelId>(id))}; result != _models.end())
  {
    return result->second.get();
  }
  else
  {
    LOG_ERROR("Ranking model not found!");
    return nullptr;
  }
}

std::vector<std::pair<TestSettings, ChartData>> ImageRanker::RunGridTest(
    const std::vector<TestSettings>& userTestsSettings)
{
  LOG_ERROR("ImageRanker::RunGridTest() NOT IMPLEMENTED");
  return std::vector<std::pair<TestSettings, ChartData>>();
  //// Final result set
  // std::vector<std::pair<TestSettings, ChartData>> results;

  // std::vector<std::thread> tPool;
  // tPool.reserve(10);

  // constexpr unsigned int numThreads{8};
  // size_t numTests{ GridTest::m_testSettings.size() };

  // size_t numTestsPerThread{(numTests / numThreads) + 1};

  // size_t from{ 0ULL };
  // size_t to{ 0ULL };

  //// Cache all queries to avoid data races
  // GetCachedQueries(QueryOriginId::cDeveloper);
  // GetCachedQueries(QueryOriginId::cPublic);

  // std::thread tProgress(&GridTest::ReportTestProgress);

  //// Start threads
  // for (size_t i{ 0ULL }; i < numThreads; ++i)
  //{
  //  to = from + numTestsPerThread;

  //  tPool.emplace_back(&ImageRanker::RunGridTestsFromTo, this, &results, from,
  //  to);

  //  from = to;
  //}

  //// Wait for threads
  // for (auto&& t : tPool)
  //{
  //  t.join();
  //}
  // tProgress.join();

  //// xoxo

  // auto cmp = [](const std::pair<size_t, size_t>& left, const
  // std::pair<size_t, size_t>& right)
  //{
  //  return left.second < right.second;
  //};

  // std::priority_queue<std::pair<size_t, size_t>,
  // std::vector<std::pair<size_t, size_t>>, decltype(cmp)> maxHeap(cmp);

  //// Choose the winner
  // size_t i{0ULL};
  // for (auto&& test : results)
  //{
  //  size_t score{0ULL};
  //  size_t j{ 0ULL };

  //  size_t numHalf{ test.second.size() / 2 };
  //  for (auto&& val : test.second)
  //  {
  //    // Only calculate first half
  //    if (j >= numHalf)
  //    {
  //      break;
  //    }
  //    score += val.second;

  //    ++j;
  //  }

  //  maxHeap.push(std::pair(i, score));

  //  ++i;
  //}

  // LOG("All tests complete.");

  // std::vector<std::pair<TestSettings, ChartData>> resultsReal;

  // for (size_t i{ 0ULL }; i < 15; ++i)
  //{
  //  auto item = maxHeap.top();
  //  maxHeap.pop();

  //  resultsReal.emplace_back(results[item.first]);
  //}
  //
  // LOG("Winner models and settings selected.");

  // return resultsReal;
}

void ImageRanker::RunGridTestsFromTo(std::vector<std::pair<TestSettings, ChartData>>* pDest, size_t fromIndex,
                                     size_t toIndex)
{
  LOG_ERROR("Not implemented : RunGridTestsFromTo()!");
  return;

  //// If to index out of bounds
  // if (toIndex > GridTest::m_testSettings.size())
  //{
  //  toIndex = GridTest::m_testSettings.size();
  //}

  // auto to = GridTest::m_testSettings.begin() + toIndex;

  //// Itarate over that interval
  // size_t i{0ULL};
  // for (auto it = GridTest::m_testSettings.begin() + fromIndex; it != to;
  // ++it)
  //{
  //  auto testSet{ (*it) };

  //  // Run test
  //  auto chartData{ RunModelTestWrapper(std::get<0>(testSet),
  //  std::get<1>(testSet), std::get<2>(testSet), std::vector<std::string>({
  //  "1"s }), std::get<3>(testSet), std::get<4>(testSet)) };

  //  pDest->emplace_back(testSet, std::move(chartData));

  //  GridTest::ProgressCallback();

  //  ++i;
  //}
}

std::tuple<const Image*, bool, size_t> ImageRanker::GetCoupledImagesNative() const
{
  // Get Viret KW data
  auto queriesViret{GetCachedQueries(std::tuple(eVocabularyId::VIRET_1200_WORDNET_2019, eScoringsId::NASNET_2019),
                                     UserDataSourceId::cAll)};

  std::vector<std::tuple<size_t, std::string, size_t, std::string, bool>> queriesNative{
      GetUserAnnotationNativeQueriesCached()};

  // image ID -> (Number of annotations left for this ID, Number of them without
  // examples)
  std::map<size_t, std::pair<size_t, size_t>> imageIdOccuranceMap;

  size_t totalCounter{0_z};

  // Add pluses there first
  for (auto&& v : queriesViret)
  {
    auto&& [imageId, fml, withExamples] = v[0];

    auto countRes{imageIdOccuranceMap.count(imageId)};

    // If not existent Image ID add it
    if (countRes <= 0)
    {
      imageIdOccuranceMap.emplace(imageId, std::pair{0_z, 0_z});
    }

    // Increment count
    ++imageIdOccuranceMap[imageId].first;
    ++totalCounter;

    // If should be without examples
    if (!withExamples)
    {
      ++imageIdOccuranceMap[imageId].second;
    }
  }

  // Subtract already created Google mathing ones
  for (auto&& [imageId, query, timestamp, sessionId, isManuallyValidated] : queriesNative)
  {
    auto countRes{imageIdOccuranceMap.count(imageId)};

    // Skip Google ones that are not created for viret
    if (countRes <= 0)
    {
      continue;
    }

    // Increment count
    auto& i{imageIdOccuranceMap[imageId].first};
    auto& ii{imageIdOccuranceMap[imageId].second};

    if (i > 0)
    {
      --i;
      --totalCounter;
    }

    // If this record paired
    if (i <= 0)
    {
      imageIdOccuranceMap.erase(imageId);
    }
  }

  // Get random item from map
  auto it = imageIdOccuranceMap.begin();

  if (totalCounter <= 0)
  {
    return std::tuple(GetRandomImage(), true, 0_z);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0_z, totalCounter - 1_z);

  std::advance(it, dis(gen));

  auto pImg{GetImageDataById(it->first)};

  return std::tuple(pImg, true, totalCounter);
}

std::tuple<const Image*, bool, size_t> ImageRanker::GetCouplingImage() const
{
  // Get Viret KW data
  auto queriesViret{GetCachedQueries(std::tuple(eVocabularyId::VIRET_1200_WORDNET_2019, eScoringsId::NASNET_2019),
                                     UserDataSourceId::cAll)};

  auto queriesGoogle{GetCachedQueries(std::tuple(eVocabularyId::GOOGLE_AI_20K_2019, eScoringsId::GOOGLE_AI_2019),
                                      UserDataSourceId::cAll)};

  // image ID -> (Number of annotations left for this ID, Number of them without
  // examples)
  std::map<size_t, std::pair<size_t, size_t>> imageIdOccuranceMap;

  size_t totalCounter{0_z};

  // Add pluses there first
  for (auto&& v : queriesViret)
  {
    auto&& [imageId, fml, withExamples] = v[0];

    auto countRes{imageIdOccuranceMap.count(imageId)};

    // If not existent Image ID add it
    if (countRes <= 0)
    {
      imageIdOccuranceMap.emplace(imageId, std::pair{0_z, 0_z});
    }

    // Increment count
    ++imageIdOccuranceMap[imageId].first;
    ++totalCounter;

    // If should be without examples
    if (!withExamples)
    {
      ++imageIdOccuranceMap[imageId].second;
    }
  }

  // Subtract already created Google mathing ones
  for (auto&& v : queriesGoogle)
  {
    auto&& [imageId, fml, withExamples] = v[0];

    auto countRes{imageIdOccuranceMap.count(imageId)};

    // Skip Google ones that are not created for viret
    if (countRes <= 0)
    {
      continue;
    }

    // Increment count
    auto& i{imageIdOccuranceMap[imageId].first};
    auto& ii{imageIdOccuranceMap[imageId].second};

    // If without examples
    if (!withExamples)
    {
      if (i > 0)
      {
        if (ii > 0)
        {
          --i;
          --ii;
          --totalCounter;
        }
        else
        {
          continue;
        }
      }
    }
    else
    {
      if (i > 0)
      {
        --i;
        --totalCounter;
      }
    }

    // If this record paired
    if (i <= 0)
    {
      imageIdOccuranceMap.erase(imageId);
    }
  }

  // Get random item from map
  auto it = imageIdOccuranceMap.begin();

  if (totalCounter <= 0)
  {
    return std::tuple(GetRandomImage(), true, 0_z);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0_z, totalCounter - 1_z);

  std::advance(it, dis(gen));

  auto pImg{GetImageDataById(it->first)};

  return std::tuple(pImg, !((bool)it->second.second), totalCounter);
}


NearKeywordsResponse ImageRanker::GetNearKeywords(DataId data_ID, const std::string& prefix,
                                                  size_t numResults, bool withExampleImages)
{
  // Force lowercase
  std::locale loc;
  std::string lower;

  // Convert to lowercase
  for (auto elem : prefix)
  {
    lower.push_back(std::tolower(elem, loc));
  }

  KeywordsContainer* pkws{nullptr};

  // Save shortcuts
  switch (std::get<0>(data_ID))
  {
    case eVocabularyId::VIRET_1200_WORDNET_2019:
      pkws = _pViretKws;
      break;

    case eVocabularyId::GOOGLE_AI_20K_2019:
      pkws = _pGoogleKws;
      break;
  }

  auto suggestedKeywordsPtrs{pkws->GetNearKeywordsPtrs(lower, numResults)};

  if (withExampleImages)
  {
    // Load representative images for keywords
    for (auto&& pKw : suggestedKeywordsPtrs)
    {
      LoadRepresentativeImages(data_ID, pKw);
    }
  }

  return suggestedKeywordsPtrs;
}

bool ImageRanker::LoadRepresentativeImages(DataId data_ID, Keyword* pKw)
{
  // If examples already loaded
  if (!pKw->m_exampleImageFilenames.empty())
  {
    return true;
  }

  std::vector<std::string> queries;
  queries.push_back(std::to_string(pKw->m_wordnetId));

  // Get first results for this keyword
  std::vector<const Image*> relevantImages{
      std::get<0>(GetRelevantImages(data_ID, queries, 10, DEFAULT_AGG_FUNCTION, DEFAULT_RANKING_MODEL,
                                    DEFAULT_MODEL_SETTINGS, DEFAULT_TRANSFORM_SETTINGS))};

  // Push those into examples
  for (auto pImg : relevantImages)
  {
    pKw->m_exampleImageFilenames.emplace_back(pImg->m_filename);
  }

  return true;
}

Keyword* ImageRanker::GetKeywordByVectorIndex(DataId data_ID, size_t index)
{
  return GetCorrectKwContainerPtr(data_ID)->GetKeywordPtrByVectorIndex(index);
}

const Keyword* ImageRanker::GetKeywordConstPtr(eVocabularyId kwDataType, size_t keywordId) const
{
  const KeywordsContainer* kwContPtr{GetCorrectKwContainerPtr(DataId{kwDataType, eScoringsId::NASNET_2019})};

  if (!kwContPtr)
  {
    return nullptr;
  }

  return kwContPtr->GetKeywordConstPtrByWordnetId(keywordId);
}

Keyword* ImageRanker::GetKeywordPtr(eVocabularyId kwDataType, size_t keywordId)
{
  auto kwContPtr{GetCorrectKwContainerPtr(DataId{kwDataType, eScoringsId::NASNET_2019})};

  if (!kwContPtr)
  {
    return nullptr;
  }

  return kwContPtr->GetKeywordPtrByWordnetId(keywordId);
}

std::vector<std::string> ImageRanker::GetImageFilenamesTrecvid() const
{
  // If no map file provided, use directory layout
  if (_imageToIdMapFilepath.empty())
  {
    LOG_ERROR("Image filename file not provided.");
    return std::vector<std::string>();
  }

  // Open file with list of files in images dir
  std::ifstream inFile(_imageToIdMapFilepath, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    LOG_ERROR(std::string("Error opening file :") + _imageToIdMapFilepath);
  }

  std::vector<std::string> result;

  std::string line;

  size_t i{0_z};

  // While there are lines in file
  while (std::getline(inFile, line))
  {
    // auto [videoId, shotId, frameNumber] { ParseVideoFilename(line) };

    line = line.substr(6, line.length() - 6);

    result.emplace_back(line);

    ++i;
  }

  return result;
}

std::tuple<std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
           std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
           std::vector<std::pair<size_t, std::string>>>
ImageRanker::GetImageKeywordsForInteractiveSearch(size_t imageId, size_t numResults, DataId data_ID,
                                                  bool withExampleImages)
{
  std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>> hypernyms;
  std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>> nonHypernyms;
  nonHypernyms.reserve(numResults);

  auto img{GetImageDataById(imageId)};
  if (img == nullptr)
  {
    LOG_ERROR("Image not found.");
    return std::tuple<std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
                      std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
                      std::vector<std::pair<size_t, std::string>>>();
  }

  // Get kws
  size_t i{0ULL};
  for (auto&& [kwPtr, kwScore] : img->_topKeywords[data_ID])
  {
    if (!(i < numResults))
    {
      break;
    }

    std::string word{kwPtr->m_word};
    std::vector<std::string> exampleImagesFilepaths;

    // Get example images
    if (withExampleImages)
    {
      LoadRepresentativeImages(data_ID, kwPtr);

      exampleImagesFilepaths = kwPtr->m_exampleImageFilenames;
    }

    nonHypernyms.emplace_back(
        std::tuple(kwPtr->m_wordnetId, std::move(word), kwScore, std::move(exampleImagesFilepaths)));

    ++i;
  }

  // Get video/shot images
  std::vector<std::pair<size_t, std::string>> succs;

  size_t numSucc{img->m_numSuccessorFrames};
  for (size_t i{0_z}; i <= numSucc; ++i)
  {
    size_t nextId{img->m_imageId + (i * _imageIdStride)};

    auto pImg{GetImageDataById(nextId)};

    succs.emplace_back(nextId, pImg->m_filename);
  }

  return std::tuple(std::move(hypernyms), std::move(nonHypernyms), std::move(succs));
}

void ImageRanker::LowMem_RecalculateHypernymsInVectorUsingMax(std::vector<float>& binVectorRef)
{
  // Iterate over all bins in this vector
  for (auto&& [it, i]{std::tuple(binVectorRef.begin(), size_t{0})}; it != binVectorRef.end(); ++it, ++i)
  {
    auto&& bin{*it};

    auto pKw{_pViretKws->GetKeywordConstPtrByVectorIndex(i)};

    float binValue{0.0f};

    // Iterate over all indices this keyword interjoins
    for (auto&& kwIndex : pKw->m_hyponymBinIndices)
    {
      binValue = std::max(binValue, binVectorRef[kwIndex]);
    }

    bin = binValue;
  }
}



std::vector<GameSessionQueryResult> ImageRanker::SubmitUserQueriesWithResults(
    DataId data_ID, std::vector<GameSessionInputQuery> inputQueries, UserDataSourceId origin)
{
  /******************************
    Save query to database
  *******************************/
  // Input format:
  // <SessionID, ImageID, User query - "wID1&wID1& ... &wIDn">

  // Resolve query origin
  size_t originNumber{static_cast<size_t>(origin)};

  // Store it into database
  std::string sqlQuery{
      "INSERT INTO `queries` (query, keyword_data_type, scoring_data_type, "
      "image_id, type, sessionId) VALUES "};

  for (auto&& query : inputQueries)
  {
    // Get image ID
    size_t imageId = std::get<1>(query);
    std::string queryString = std::get<2>(query);

    sqlQuery += "('"s + EncodeAndQuery(queryString) + "', "s + std::to_string((size_t)std::get<0>(data_ID)) + ", "s +
                std::to_string((size_t)std::get<1>(data_ID)) + ", " + std::to_string(imageId) + ", "s +
                std::to_string(originNumber) + ", '"s + std::get<0>(query) + "'),"s;
  }

  sqlQuery.pop_back();
  sqlQuery += ";";

  auto result = _db.NoResultQuery(sqlQuery);
  if (result != 0)
  {
    LOG_ERROR(std::string("query: ") + sqlQuery +
              std::string("\n\nInserting queries into DB failed with error code: ") + std::to_string(result));
  }

  /******************************
    Construct result for user
  *******************************/
  std::vector<GameSessionQueryResult> userResult;
  userResult.reserve(inputQueries.size());

  for (auto&& query : inputQueries)
  {
    // Get user keywords tokens
    std::vector<std::string> userKeywords{StringenizeAndQuery(data_ID, std::get<2>(query))};

    // Get image ID
    size_t imageId = std::get<1>(query);

    // Get image filename
    std::string imageFilename{GetImageFilenameById(imageId)};

    std::vector<std::pair<std::string, float>> netKeywordsProbs{};

    userResult.emplace_back(std::get<0>(query), std::move(imageFilename), std::move(userKeywords),
                            GetHighestProbKeywords(data_ID, imageId, 10ULL));
  }

  return userResult;
}

void ImageRanker::SubmitUserDataNativeQueries(std::vector<std::tuple<size_t, std::string, std::string>>& queries)
{
  // Store it into database
  std::string sqlQuery{
      "INSERT INTO `user_data_native_queries` \
      (image_id, query, session_id) \
    VALUES "};

  for (auto&& [imageId, query, sessionId] : queries)
  {
    sqlQuery += "("s + std::to_string(imageId) + ", '"s + query + "', '"s + sessionId + "'),"s;
  }

  sqlQuery.pop_back();
  sqlQuery += ";";

  auto result = _db.NoResultQuery(sqlQuery);
  if (result != 0)
  {
    LOG_ERROR(std::string("query: ") + sqlQuery +
              std::string("\n\nInserting queries into DB failed with error code: ") + std::to_string(result));
  }
}

size_t ImageRanker::MapIdToVectorIndex(size_t id) const { return id / _imageIdStride; }

std::vector<std::pair<std::string, float>> ImageRanker::GetHighestProbKeywords(DataId data_ID,
                                                                               size_t imageId, size_t N) const
{
  N = std::min(N, NUM_TOP_KEYWORDS);

  // Find image in map
  Image* pImg = _images[MapIdToVectorIndex(imageId)].get();

  // Construct new subvector
  std::vector<std::pair<std::string, float>> result;
  result.reserve(N);

  auto kwScorePairs = pImg->_topKeywords.at(data_ID);

  // Get first N highest probabilites
  for (size_t i = 0ULL; i < N; ++i)
  {
    if (i >= kwScorePairs.size()) break;

    float kwScore{std::get<1>(kwScorePairs[i])};

    // Get keyword string
    std::string keyword{std::get<0>(kwScorePairs[i])->m_word};
    ;

    // Place it into result vector
    result.emplace_back(std::pair(keyword, kwScore));
  }

  return result;
}

std::vector<std::string> ImageRanker::TokenizeAndQuery(std::string_view query) const
{
  // Create sstram from query
  std::stringstream querySs{query.data()};

  std::vector<std::string> resultTokens;
  std::string tokenString;

  while (std::getline(querySs, tokenString, '&'))
  {
    // If empty string
    if (tokenString.empty())
    {
      continue;
    }

    // Push new token into result
    resultTokens.emplace_back(std::move(tokenString));
  }

  return resultTokens;
}



std::vector<std::vector<UserImgQuery>> ImageRanker::DoQueryAndExpansion(
    DataId data_ID, const std::vector<std::vector<UserImgQuery>>& origQuery, size_t setting) const
{
  std::vector<std::vector<UserImgQuery>> result{origQuery};

  // Augment result query
  for (auto&& query : result)
  {
    auto& q{query[0]};

    // std::vector<std::vector<std::pair<bool, size_t>>>
    auto ccnf = std::get<1>(q);
    auto& cnf = std::get<1>(q);

#if LOG_QUERY_EXPANSION
    std::cout << "-----" << std::endl;
    std::cout << "Original:" << std::endl;
    std::cout << _pGoogleKws->cnfFormulaToString(ccnf) << std::endl;
#endif

    for (auto&& clause : ccnf)
    {
      auto kwId{clause[0].second};

      auto pKw{GetCorrectKwContainerPtr(data_ID)->GetKeywordConstPtrByVectorIndex(kwId)};

      decltype(pKw->m_expanded1Concat) concats;
      decltype(pKw->m_expanded1Concat) substrings;

      if (setting == 1)
      {
        concats = pKw->m_expanded1Concat;
        substrings = pKw->m_expanded1Substrings;
      }
      else if (setting == 2)
      {
        concats = pKw->m_expanded2Concat;
        substrings = pKw->m_expanded2Substrings;
      }

      for (auto&& pKw1 : concats)
      {
        // Copyu clasuse
        auto newClause = clause;

        newClause[0].second = pKw1->m_vectorIndex;

        cnf.push_back(std::move(newClause));
      }

      for (auto&& pKw1 : substrings)
      {
        // Copyu clasuse
        auto newClause = clause;

        newClause[0].second = pKw1->m_vectorIndex;

        cnf.push_back(std::move(newClause));
      }
    }

#if LOG_QUERY_EXPANSION
    std::cout << "New:" << std::endl;
    std::cout << GetCorrectKwContainerPtr(data_ID)->cnfFormulaToString(cnf) << std::endl;
#endif
  }

  return result;
}

std::vector<std::vector<UserImgQuery>> ImageRanker::DoQueryOrExpansion(
    DataId data_ID, const std::vector<std::vector<UserImgQuery>>& origQuery, size_t setting) const
{
  std::vector<std::vector<UserImgQuery>> result{origQuery};

  if (setting == 0)
  {
    return result;
  }

  // Augment result query
  for (auto&& query : result)
  {
    auto& q{query[0]};

    // std::vector<std::vector<std::pair<bool, size_t>>>
    auto ccnf = std::get<1>(q);
    auto& cnf = std::get<1>(q);

#if LOG_QUERY_EXPANSION
    std::cout << "-----" << std::endl;
    std::cout << GetCorrectKwContainerPtr(data_ID)->cnfFormulaToString(ccnf) << std::endl;
#endif

    for (auto&& clause : cnf)
    {
      std::set<size_t> ids;

      for (auto&& [a, b] : clause)
      {
        ids.insert(b);
      }

      for (auto&& [a, b] : clause)
      {
        auto kwId{b};

        auto pKw{GetCorrectKwContainerPtr(data_ID)->GetKeywordConstPtrByVectorIndex(kwId)};

        if (!pKw)
        {
          continue;
        }

        if (setting == 1 || setting == 2 || setting == 23 || setting == 13)
        {
          decltype(pKw->m_expanded1Concat) concats;
          decltype(pKw->m_expanded1Substrings) substrings;

          if (setting == 1 || setting == 13)
          {
            concats = pKw->m_expanded1Concat;
            substrings = pKw->m_expanded1Substrings;
          }
          else if (setting == 2 || setting == 23)
          {
            concats = pKw->m_expanded2Concat;
            substrings = pKw->m_expanded2Substrings;
          }

          for (auto&& pKw1 : concats)
          {
            if (ids.insert(pKw1->m_vectorIndex).second)
            {
              clause.emplace_back(false, pKw1->m_vectorIndex);
            }
          }

          for (auto&& pKw1 : substrings)
          {
            if (ids.insert(pKw1->m_vectorIndex).second)
            {
              clause.emplace_back(false, pKw1->m_vectorIndex);
            }
          }
        }
        else if (setting == 3 || setting == 23 || setting == 13)
        {
          // word2vec

          auto expSet = pKw->m_wordToVec;

          for (auto&& [pKw1, dist] : expSet)
          {
            if (dist >= W2V_DISTANCE_THRESHOLD)
            {
              if (ids.insert(pKw1->m_vectorIndex).second)
              {
                clause.emplace_back(false, pKw1->m_vectorIndex);
              }
            }
          }
        }
        else
        {
          LOG_ERROR("Somthing went wrong");
        }
      }
    }
#if LOG_QUERY_EXPANSION
    std::cout << "=> " << GetCorrectKwContainerPtr(data_ID)->cnfFormulaToString(cnf) << std::endl;

#endif
  }

  return result;
}

ChartData ImageRanker::RunModelTestWrapper(DataId data_ID, InputDataTransformId aggId,
                                           RankingModelId modelId, UserDataSourceId dataSource,
                                           const SimulatedUserSettings& simulatedUserSettings,
                                           const RankingModelSettings& aggModelSettings,
                                           const InputDataTransformSettings& netDataTransformSettings,
                                           size_t expansionSettings) const
{
  std::vector<std::vector<UserImgQuery>> testQueries;

  // If data source should be simulated
  if (static_cast<int>(dataSource) >= SIMULATED_QUERIES_ENUM_OFSET)
  {
    // Parse simulated user settings
    auto simUser{GetSimUserSettings(simulatedUserSettings)};

    if (static_cast<int>(dataSource) >= USER_PLUS_SIMULATED_QUERIES_ENUM_OFSET)
    {
      // xoxo
      //
      // Generate temporal queries for real user queries
      //

      // Get real user queries with simulated queries added
      testQueries = GetExtendedRealQueries(data_ID, dataSource, simUser);
    }
    else
    {
      // Get simulated queries
      // testQueries = GetSimulatedQueries(data_ID, dataSource, simUser);// testQueries =
      // GetSimulatedQueries(data_ID, dataSource, simUser);// testQueries = GetSimulatedQueries(data_ID,
      // dataSource, simUser);// testQueries = GetSimulatedQueries(data_ID, dataSource, simUser);// testQueries =
      // GetSimulatedQueries(data_ID, dataSource, simUser);// testQueries = GetSimulatedQueries(data_ID,
      // dataSource, simUser);
      testQueries = GetSimulatedQueries(data_ID, 20000_z, false, simUser);

#if GENERATE_SIMULATED_USER_QUERIES_JSON

      json j;
      j.array();
      // std::tuple<size_t, CnfFormula, bool>
      for (auto&& aaa : testQueries)
      {
        auto [imageId, cnfFormula, y]{aaa[0]};

        json item;
        json itemArr;
        itemArr.array();

        item["ImageId"] = imageId * 50;
        // std::vector<std::pair<bool, size_t>>
        for (auto&& clauses : cnfFormula)
        {
          for (auto&& [x, id] : clauses)
          {
            itemArr += id;
          }
        }
        item["KwIds"] = itemArr;

        j += item;
      }

      std::string s = j.dump(4);

      std::cout << s << std::endl;
#endif
    }
  }
  else
  {
    // Get queries
    testQueries = GetCachedQueries(data_ID, dataSource);
  }

#if DO_SUBSTRING_EXPANSION

#if SUBSTRING_EXPANSION_TYPE == 0

  auto testQueriesExpanded{DoQueryAndExpansion(data_ID, testQueries, expansionSettings)};

#elif SUBSTRING_EXPANSION_TYPE == 1

  auto testQueriesExpanded{DoQueryOrExpansion(data_ID, testQueries, expansionSettings)};

#endif

#else

  auto testQueriesExpanded{testQueries};

#endif

  // Get desired transformation
  auto pNetDataTransformFn = GetAggregationById(aggId);
  // Setup transformation correctly
  pNetDataTransformFn->SetTransformationSettings(netDataTransformSettings);

  // Get disired model
  auto pRankingModel = GetRankingModelById(modelId);
  // Setup model correctly
  pRankingModel->SetModelSettings(aggModelSettings);

  // Run test
  return pRankingModel->RunModelTestWithOrigQueries(data_ID, pNetDataTransformFn, &_indexKwFrequency,
                                                    testQueriesExpanded, testQueries, _images, _keywordContainers);
}

std::vector<ChartData> ImageRanker::RunModelSimulatedQueries(std::string run_name, DataId data_ID,
                                                             InputDataTransformId aggId, RankingModelId modelId,
                                                             UserDataSourceId dataSource,
                                                             const SimulatedUserSettings& simulatedUserSettings,
                                                             const RankingModelSettings& aggModelSettings,
                                                             const InputDataTransformSettings& netDataTransformSettings,
                                                             size_t expansionSettings) const
{
  std::vector<std::vector<std::vector<UserImgQuery>>> testQueriesVecs;
  std::vector<ChartData> chartsData;

  for (size_t i = 1; i < 7; ++i)
  {
    std::vector<std::vector<UserImgQuery>> testQueries;

    SimulatedUser simUser;
    simUser.m_exponent = i;

    // Get simulated queries
    testQueries = GetSimulatedQueries(data_ID, 20000_z, false, simUser);
    testQueriesVecs.emplace_back(std::move(testQueries));
  }

  for (auto&& testQueries : testQueriesVecs)
  {
#if 0

  json j;
  j.array();
  //std::tuple<size_t, CnfFormula, bool>
  for (auto&& aaa : testQueries) {
    auto [imageId, cnfFormula, y]{aaa[0]};

    json item;
    json itemArr;
    itemArr.array();

    item["ImageId"] = imageId * 50;
    // std::vector<std::pair<bool, size_t>>
    for (auto&& clauses : cnfFormula) {
      for (auto&& [x, id] : clauses) {
        itemArr += id;
      }
    }
    item["KwIds"] = itemArr;

    j += item;
  }

  std::string s = j.dump(4);

  std::cout << s << std::endl;
#endif

    auto testQueriesExpanded{testQueries};

    // Get desired transformation
    auto pNetDataTransformFn = GetAggregationById(aggId);
    // Setup transformation correctly
    pNetDataTransformFn->SetTransformationSettings(netDataTransformSettings);

    // Get disired model
    auto pRankingModel = GetRankingModelById(modelId);
    // Setup model correctly
    pRankingModel->SetModelSettings(aggModelSettings);

    // Run test
    auto charts =
        pRankingModel->RunModelTestWithOrigQueries(data_ID, pNetDataTransformFn, &_indexKwFrequency,
                                                   testQueriesExpanded, testQueries, _images, _keywordContainers);
    chartsData.emplace_back(std::move(charts));
    std::cout << "+++" << std::endl;
  }

  // now let's put that into the file
  std::ofstream fs_out_file(run_name + ".csv");

  if (!fs_out_file.is_open()) throw std::runtime_error("AAAA");

  for (size_t jj = 0; jj < vec_of_ranks[0].size(); ++jj)
  {
    for (size_t i = 0; i < 6; ++i)
    {
      fs_out_file << vec_of_ranks[i][jj] << ";";
    }
    fs_out_file << std::endl;
  }
  fs_out_file.close();

  vec_of_ranks.clear();

  return chartsData;
}

UserImgQuery ImageRanker::GetSimulatedQueryForImage(size_t imageId, const SimulatedUser& simUser) const
{
  constexpr size_t from{2_z};
  constexpr size_t to{6_z};

  auto pImgData{GetImageDataById(imageId)};

  const auto& linBinVector{
      pImgData->_rawSimUserData.at(std::tuple(eVocabularyId::VIRET_1200_WORDNET_2019, eScoringsId::NASNET_2019))};

  // Calculate transformed vector
  float totalSum{0.0f};
  std::vector<float> transformedData;
  for (auto&& value : linBinVector)
  {
    float newValue{pow(value, simUser.m_exponent)};
    transformedData.push_back(newValue);

    totalSum += newValue;
  }

  // Get scaling coef
  float scaleCoef{1 / totalSum};

  // Normalize
  size_t i{0_z};
  float cummulSum{0.0f};
  for (auto&& value : transformedData)
  {
    cummulSum += value * scaleCoef;

    transformedData[i] = cummulSum;

    ++i;
  }

  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  auto randLabel{static_cast<float>(distribution(generator))};

  size_t numLabels{static_cast<size_t>((randLabel * (to - from)) + from)};
  std::vector<size_t> queryLabels;
  for (size_t i{0_z}; i < numLabels; ++i)
  {
    // Get random number between [0, 1] from uniform distribution
    float rand{static_cast<float>(distribution(generator))};
    size_t labelIndex{0};

    // Iterate through discrete points while we haven't found correct point
    while (transformedData[labelIndex] < rand)
    {
      ++labelIndex;
    }

    queryLabels.push_back(labelIndex);
  }

  std::vector<Clause> queryFormula;
  queryFormula.reserve(numLabels);

  // Create final formula with wordnet IDs
  for (auto&& index : queryLabels)
  {
    auto a = _pViretKws->GetKeywordPtrByVectorIndex(index);
    Clause meta;
    meta.emplace_back(false, a->m_vectorIndex);

    queryFormula.emplace_back(std::move(meta));
  }

  return std::tuple(imageId, queryFormula, true);
}

std::vector<std::vector<UserImgQuery>> ImageRanker::GetSimulatedQueries(DataId data_ID,
                                                                        UserDataSourceId dataSource,
                                                                        const SimulatedUser& simUser) const
{
  // Determine what id would that be if not simulated user
  UserDataSourceId dataSourceNotSimulated{
      static_cast<UserDataSourceId>(static_cast<int>(dataSource) - SIMULATED_QUERIES_ENUM_OFSET)};

  // Fetch real user queries to immitate them
  std::vector<std::vector<UserImgQuery>> realUsersQueries{GetCachedQueries(data_ID, dataSourceNotSimulated)};

  // Prepare result structure
  std::vector<std::vector<UserImgQuery>> resultSimulatedQueries;
  // Reserve space properly
  resultSimulatedQueries.reserve(realUsersQueries.size());

  for (auto&& queries : realUsersQueries)
  {
    std::vector<UserImgQuery> singleQuery;

    for (auto&& [imageId, formula, withExamples] : queries)
    {
      auto simulatedQuery{GetSimulatedQueryForImage(imageId, simUser)};

      singleQuery.push_back(std::move(simulatedQuery));
    }
    resultSimulatedQueries.push_back(std::move(singleQuery));
  }

  return resultSimulatedQueries;
}

std::vector<std::vector<UserImgQuery>> ImageRanker::GetSimulatedQueries(DataId data_ID, size_t num_quries,
                                                                        bool sample_targets,
                                                                        const SimulatedUser& simUser) const
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0_z, _images.size());

  std::vector<std::vector<UserImgQuery>> result_queries;
  // Reserve space properly
  result_queries.reserve(num_quries);
  for (size_t i = 0_z; i < num_quries; ++i)
  {
    size_t img_ID = i;

    if (sample_targets)
    {
      img_ID = dis(gen);
    }
    auto simulatedQuery{GetSimulatedQueryForImage(img_ID, simUser)};

    std::vector<UserImgQuery> n;
    n.emplace_back(std::move(simulatedQuery));

    result_queries.push_back(std::move(n));
  }

  return result_queries;
}

std::vector<std::vector<UserImgQuery>> ImageRanker::GetExtendedRealQueries(DataId data_ID,
                                                                           UserDataSourceId dataSource,
                                                                           const SimulatedUser& simUser) const
{
  // Determine what id would that be if not simulated user
  UserDataSourceId dataSourceNotSimulated{
      static_cast<UserDataSourceId>(static_cast<int>(dataSource) - USER_PLUS_SIMULATED_QUERIES_ENUM_OFSET)};

  // Fetch real user queries to immitate them
  std::vector<std::vector<UserImgQuery>> realUsersQueries{GetCachedQueries(data_ID, dataSourceNotSimulated)};

  // Prepare result structure - copy of real user queries
  std::vector<std::vector<UserImgQuery>> resultSimulatedQueries{realUsersQueries};

  std::random_device rd;   // only used once to initialise (seed) engine
  std::mt19937 rng(rd());  // random-number engine used (Mersenne-Twister in this case)

  size_t iterator{0_z};
  for (auto&& queries : realUsersQueries)
  {
    for (auto&& [imageId, formula, withExamples] : queries)
    {
      auto imgIt{_images.begin() + MapIdToVectorIndex(imageId)};

      if (imgIt == _images.end())
      {
        LOG_ERROR("aaa");
      }

      // Get image ptr
      Image* pImg{imgIt->get()};

      size_t numSuccs{pImg->m_numSuccessorFrames};
      if (numSuccs <= 0)
      {
        break;
      }
      // Get how much we will offset from this image
      std::uniform_int_distribution<size_t> uni(1, std::min((size_t)MAX_TEMP_QUERY_OFFSET, numSuccs));
      size_t offset{uni(rng)};

      // Offset iterator
      for (size_t i{0_z}; i < offset; ++i)
      {
        ++imgIt;
      }

      auto simulatedQuery{GetSimulatedQueryForImage(pImg->m_imageId, simUser)};

      resultSimulatedQueries[iterator].push_back(std::move(simulatedQuery));

      // Only the first one
      break;
    }

    ++iterator;
  }

  return resultSimulatedQueries;
}

std::vector<UserImgQueryRaw>& ImageRanker::GetCachedQueriesRaw(UserDataSourceId dataSource) const
{
  static std::vector<UserImgQueryRaw> cachedData0;
  static std::vector<UserImgQueryRaw> cachedData1;
  static std::vector<UserImgQueryRaw> cachedData2;

  static std::chrono::steady_clock::time_point cachedData0Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData1Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData2Ts = std::chrono::steady_clock::now();

  auto currentTime = std::chrono::steady_clock::now();

  switch (dataSource)
  {
    case UserDataSourceId::cDeveloper:

      if (cachedData0.empty() || cachedData0Ts < currentTime)
      {
        cachedData0.clear();

        // Fetch pairs of <Q, Img>
        std::string query(
            "SELECT image_id, query FROM "
            "`image-ranker-collector-data2`.queries WHERE type = " +
            std::to_string((int)dataSource) + ";");

        auto dbData = _db.ResultQuery(query);

        if (dbData.first != 0)
        {
          LOG_ERROR("Error getting queries from database."s);
        }

        // Parse DB results
        for (auto&& idQueryRow : dbData.second)
        {
          size_t imageId{static_cast<size_t>(strToInt(idQueryRow[0].data()))};
          std::vector<size_t> queryWordnetIds{_pViretKws->GetCanonicalQueryNoRecur(idQueryRow[1])};

          cachedData0.emplace_back(std::move(imageId), std::move(queryWordnetIds));
        }

        cachedData0Ts = std::chrono::steady_clock::now();
        cachedData0Ts += std::chrono::seconds(QUERIES_CACHE_LIFETIME);
      }

      return cachedData0;

      break;

    case UserDataSourceId::cPublic:

      if (cachedData1.empty() || cachedData1Ts < currentTime)
      {
        cachedData1.clear();

        // Fetch pairs of <Q, Img>
        std::string query(
            "SELECT image_id, query FROM "
            "`image-ranker-collector-data2`.queries WHERE type = " +
            std::to_string((int)dataSource) + ";");
        auto dbData = _db.ResultQuery(query);

        if (dbData.first != 0)
        {
          LOG_ERROR("Error getting queries from database."s);
        }

        // Parse DB results
        for (auto&& idQueryRow : dbData.second)
        {
          size_t imageId{static_cast<size_t>(strToInt(idQueryRow[0].data()))};
          std::vector<size_t> queryWordnetIds{_pViretKws->GetCanonicalQueryNoRecur(idQueryRow[1])};

          cachedData1.emplace_back(std::move(imageId), std::move(queryWordnetIds));
        }

        cachedData1Ts = std::chrono::steady_clock::now();
        cachedData1Ts += std::chrono::seconds(QUERIES_CACHE_LIFETIME);
      }

      return cachedData1;

      break;
  }

  return cachedData0;
}

std::vector<UserDataNativeQuery>& ImageRanker::GetUserAnnotationNativeQueriesCached() const
{
  static std::vector<UserDataNativeQuery> cachedData0;
  static std::chrono::steady_clock::time_point cachedData0Ts = std::chrono::steady_clock::now();

  auto currentTime = std::chrono::steady_clock::now();

  if (cachedData0.empty() || cachedData0Ts < currentTime)
  {
    cachedData0.clear();

    std::string sqlQuery(
        "SELECT image_id, query, created,\
        session_id, manually_validated\
      FROM `" +
        _db.GetDbName() + "`.user_data_native_queries;");

    auto dbData = _db.ResultQuery(sqlQuery);

    if (dbData.first != 0)
    {
      LOG_ERROR("Error getting queries from database."s);
    }

    // Parse DB results
    for (auto&& idQueryRow : dbData.second)
    {
      size_t imageId{static_cast<size_t>(strToInt(idQueryRow[0].data()))};
      std::string userQuery{idQueryRow[1]};
      size_t timestamp{static_cast<size_t>(strToInt(idQueryRow[2].data()))};
      std::string sessionId{idQueryRow[3]};
      bool isManuallyValidated{static_cast<bool>(strToInt(idQueryRow[2].data()))};

      cachedData0.emplace_back(imageId, std::move(userQuery), timestamp, std::move(sessionId), isManuallyValidated);
    }

    cachedData0Ts = std::chrono::steady_clock::now();
    cachedData0Ts += std::chrono::seconds(QUERIES_CACHE_LIFETIME);
  }

  return cachedData0;
}

std::vector<std::vector<UserImgQuery>>& ImageRanker::GetCachedQueries(DataId data_ID,
                                                                      UserDataSourceId dataSource) const
{
  static std::vector<std::vector<UserImgQuery>> cachedAllData;
  static std::vector<std::vector<UserImgQuery>> cachedData0;
  static std::vector<std::vector<UserImgQuery>> cachedData1;
  static std::vector<std::vector<UserImgQuery>> cachedData2;

  static std::chrono::steady_clock::time_point cachedAllDataTs = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData0Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData1Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData2Ts = std::chrono::steady_clock::now();

  auto currentTime = std::chrono::steady_clock::now();

  if (dataSource == UserDataSourceId::cAll)
  {
    if (cachedAllData.empty() || cachedAllDataTs < currentTime)
    {
      cachedAllData.clear();

      const auto& queries1 = GetCachedQueries(data_ID, UserDataSourceId::cDeveloper);
      const auto& queries2 = GetCachedQueries(data_ID, UserDataSourceId::cPublic);

      // Merge them
      std::copy(queries1.begin(), queries1.end(), back_inserter(cachedAllData));
      std::copy(queries2.begin(), queries2.end(), back_inserter(cachedAllData));

      cachedAllDataTs = std::chrono::steady_clock::now();
      cachedAllDataTs += std::chrono::seconds(QUERIES_CACHE_LIFETIME);
    }

    return cachedAllData;
  }

  switch (dataSource)
  {
    case UserDataSourceId::cDeveloper:

      if (cachedData0.empty() || cachedData0Ts < currentTime)
      {
        cachedData0.clear();

        // Fetch pairs of <Q, Img>
        std::string query(
            "\
        SELECT image_id, query, type FROM `" +
            _db.GetDbName() +
            "`.queries \
          WHERE ( type = " +
            std::to_string((int)dataSource) + " OR type =  " + std::to_string(((int)dataSource + 10)) +
            ") AND \
            keyword_data_type = " +
            std::to_string((int)std::get<0>(data_ID)) +
            " AND \
            scoring_data_type = " +
            std::to_string((int)std::get<1>(data_ID)) + ";");

        auto dbData = _db.ResultQuery(query);

        if (dbData.first != 0)
        {
          LOG_ERROR("Error getting queries from database."s);
        }

        // Parse DB results
        for (auto&& idQueryRow : dbData.second)
        {
          size_t imageId{static_cast<size_t>(strToInt(idQueryRow[0].data())) * TEST_QUERIES_ID_MULTIPLIER};

          CnfFormula queryFormula{GetCorrectKwContainerPtr(data_ID)
                                      ->GetCanonicalQuery(EncodeAndQuery(idQueryRow[1]), IGNORE_CONSTRUCTED_HYPERNYMS)};
          bool wasWithExamples{(bool)((strToInt(idQueryRow[2]) / 10) % 2)};

#if RUN_TESTS_ONLY_ON_NON_EMPTY_POSTREMOVE_HYPERNYM

          CnfFormula queryFormulaTest{
              GetCorrectKwContainerPtr(data_ID)->GetCanonicalQuery(EncodeAndQuery(idQueryRow[1]), true)};
          if (!queryFormulaTest.empty())
#else

          if (!queryFormula.empty())

#endif
          {
            std::vector<UserImgQuery> tmp;
            tmp.emplace_back(std::move(imageId), std::move(queryFormula), wasWithExamples);

            cachedData0.emplace_back(std::move(tmp));
          }
        }

        cachedData0Ts = std::chrono::steady_clock::now();
        cachedData0Ts += std::chrono::seconds(QUERIES_CACHE_LIFETIME);
      }

      return cachedData0;

      break;

    case UserDataSourceId::cPublic:

      if (cachedData1.empty() || cachedData1Ts < currentTime)
      {
        cachedData1.clear();

        // Fetch pairs of <Q, Img>
        std::string query(
            "\
        SELECT image_id, query, type FROM `" +
            _db.GetDbName() +
            "`.queries \
          WHERE (type = " +
            std::to_string((int)dataSource) + " OR type = " + std::to_string(((int)dataSource + 10)) +
            ") AND \
            keyword_data_type = " +
            std::to_string((int)std::get<0>(data_ID)) +
            " AND \
            scoring_data_type = " +
            std::to_string((int)std::get<1>(data_ID)) + ";");

        auto dbData = _db.ResultQuery(query);

        if (dbData.first != 0)
        {
          LOG_ERROR("Error getting queries from database."s);
        }

        // Parse DB results
        for (auto&& idQueryRow : dbData.second)
        {
          size_t imageId{static_cast<size_t>(strToInt(idQueryRow[0].data())) * TEST_QUERIES_ID_MULTIPLIER};

          CnfFormula queryFormula{GetCorrectKwContainerPtr(data_ID)
                                      ->GetCanonicalQuery(EncodeAndQuery(idQueryRow[1]), IGNORE_CONSTRUCTED_HYPERNYMS)};
          bool wasWithExamples{(bool)((strToInt(idQueryRow[2]) / 10) % 2)};

#if RUN_TESTS_ONLY_ON_NON_EMPTY_POSTREMOVE_HYPERNYM

          CnfFormula queryFormulaTest{
              GetCorrectKwContainerPtr(data_ID)->GetCanonicalQuery(EncodeAndQuery(idQueryRow[1]), true)};
          if (!queryFormulaTest.empty())
#else
          if (!queryFormula.empty())
#endif
          {
            std::vector<UserImgQuery> tmp;
            tmp.emplace_back(std::move(imageId), std::move(queryFormula), wasWithExamples);

            cachedData1.emplace_back(std::move(tmp));
          }
        }

        cachedData1Ts = std::chrono::steady_clock::now();
        cachedData1Ts += std::chrono::seconds(QUERIES_CACHE_LIFETIME);
      }

      return cachedData1;

      break;
  }

  return cachedData0;
}

void ImageRanker::SubmitInteractiveSearchSubmit(DataId data_ID, InteractiveSearchOrigin originType,
                                                size_t imageId, RankingModelId modelId,
                                                InputDataTransformId transformId,
                                                std::vector<std::string> modelSettings,
                                                std::vector<std::string> transformSettings, std::string sessionId,
                                                size_t searchSessionIndex, int endStatus, size_t sessionDuration,
                                                std::vector<InteractiveSearchAction> actions, size_t userId)
{
  size_t isEmpty{0_z};

  size_t countIn{0_z};
  size_t countOut{0_z};

  for (auto&& a : actions)
  {
    if (std::get<0>(a) == 0)
    {
      ++countOut;
    }
    else if (std::get<0>(a) == 1 || std::get<0>(a) == 2)
    {
      ++countIn;
    }
  }

  if (countIn == countOut)
  {
    isEmpty = 1_z;
  }

  std::stringstream query1Ss;
  query1Ss << "INSERT INTO `interactive_searches`(`keyword_data_type`, "
              "`scoring_data_type`, `type`, `target_image_id`, "
              "`model_id`, `transformation_id`, `model_settings`, "
              "`transformation_settings`, `session_id`, `user_id`, "
              "`search_session_index`, `end_status`, `session_duration`,`is_empty`)";
  query1Ss << "VALUES(" << std::to_string((int)std::get<0>(data_ID)) << ", "
           << std::to_string((int)std::get<1>(data_ID)) << "," << (int)originType << "," << imageId << ","
           << (int)modelId << "," << (int)transformId << ",";

  query1Ss << "\"";
  for (auto&& s : modelSettings)
  {
    query1Ss << s << ";";
  }
  query1Ss << "\"";
  query1Ss << ",";
  query1Ss << "\"";
  for (auto&& s : transformSettings)
  {
    query1Ss << s << ";";
  }
  query1Ss << "\"";
  query1Ss << ",\"" << sessionId << "\"," << userId << "," << searchSessionIndex << "," << endStatus << ","
           << sessionDuration << "," << isEmpty << ");";

  std::string query1{query1Ss.str()};
  size_t result1{_db.NoResultQuery(query1)};

  size_t id{_db.GetLastId()};

  std::stringstream query2Ss;
  query2Ss << "INSERT INTO `interactive_searches_actions`(`interactive_search_id`, "
              "`index`, `action`, `score`, `operand`)";
  query2Ss << "VALUES";
  {
    size_t i{0_z};
    for (auto&& action : actions)
    {
      query2Ss << "(" << id << "," << i << "," << std::get<0>(action) << "," << std::get<1>(action) << ","
               << std::get<2>(action) << ")";

      if (i < actions.size() - 1)
      {
        query2Ss << ",";
      }
      ++i;
    }
    query2Ss << ";";
  }

  std::string query2{query2Ss.str()};
  size_t result2{_db.NoResultQuery(query2)};

  if (result1 != 0 || result2 != 0)
  {
    LOG_ERROR("Failed to insert search session result.");
  }
}

std::tuple<UserAccuracyChartData, UserAccuracyChartData> ImageRanker::GetStatisticsUserKeywordAccuracy(
    UserDataSourceId queriesSource) const
{
  LOG_ERROR("Not implemented: GetStatisticsUserKeywordAccuracy()");
  return std::tuple<UserAccuracyChartData, UserAccuracyChartData>();
  /*
  std::vector<UserImgQueryRaw> queries;

  if (queriesSource == QueryOriginId::cAll)
  {
    queries = GetCachedQueriesRaw(QueryOriginId::cDeveloper);
    auto queries2 = GetCachedQueriesRaw(QueryOriginId::cPublic);

    // Merge them
    std::copy(queries2.begin(), queries2.end(), back_inserter(queries));
  }

  std::vector<size_t> hitsNonHyper;
  size_t hitsNonHyperTotal{0ULL};
  hitsNonHyper.resize(_pViretKws->GetNetVectorSize());

  std::vector<size_t> hitsHyper;
  size_t hitsHyperTotal{ 0ULL };
  auto pImg{ GetImageDataById(0) };
  hitsHyper.resize(pImg->m_hypernymsRankingSorted.size());

  for (auto&&[imgId, wnIds] : queries)
  {
    auto pImg{ GetImageDataById(imgId) };


    for (auto&& wordnetId : wnIds)
    {
      // Get keyword
      auto kw{ _pViretKws->GetWholeKeywordByWordnetId(wordnetId) };

      // If is non-hyper
      if (kw->m_vectorIndex != ERR_VAL<size_t>())
      {

        const auto& rankVec{ pImg->_rawImageScoringDataSorted };

        for (size_t i{ 0ULL }; i < rankVec.size(); ++i)
        {
          // If this is the word
          if (rankVec[i].first == kw->m_vectorIndex)
          {
            ++hitsNonHyper[i];
            ++hitsNonHyperTotal;
          }
        }
      }
      // Is hypernym
      else
      {
        const auto& rankVec{ pImg->m_hypernymsRankingSorted };

        for (size_t i{ 0ULL }; i < rankVec.size(); ++i)
        {
          // If this is the word
          if (rankVec[i].first == wordnetId)
          {
            ++hitsHyper[i];
            ++hitsHyperTotal;
          }
        }
      }
    }
  }

  float percNonHyper{ (float)hitsNonHyperTotal / (hitsHyperTotal +
  hitsNonHyperTotal) }; float percHyper{ 1 - percNonHyper };


  UserAccuracyChartDataMisc nonHyperMisc{ (size_t)queriesSource, percNonHyper };
  UserAccuracyChartDataMisc hyperMisc{ (size_t)queriesSource, percHyper };

  float scaleDownNonHyper{ 10 };
  float scaleDownHyper{ 10 };

  ChartData nonHyperChartData;
  ChartData hyperChartData;


  nonHyperChartData.emplace_back(0, 0);
  {
    size_t chIndex{ 1ULL };
    for (auto it = hitsNonHyper.begin(); it != hitsNonHyper.end(); )
    {
      size_t locMax{ 0ULL };
      for (size_t i{ 0ULL }; i < scaleDownNonHyper; ++i, ++it)
      {
        if (it == hitsNonHyper.end())
        {
          break;
        }

        if (*it > locMax)
        {
          locMax = *it;
        }
      }

      nonHyperChartData.emplace_back(static_cast<uint32_t>(chIndex *
  scaleDownNonHyper), static_cast<uint32_t>(locMax));
      ++chIndex;
    }
  }

  hyperChartData.emplace_back(0, 0);

  {
    size_t chIndex{ 1ULL };
    for (auto it = hitsHyper.begin(); it != hitsHyper.end();)
    {
      size_t locMax{ 0ULL };
      for (size_t i{ 0ULL }; i < scaleDownHyper; ++i, ++it)
      {
        if (it == hitsHyper.end())
        {
          break;

        }
        if (*it > locMax)
        {
          locMax = *it;
        }
      }

      hyperChartData.emplace_back(static_cast<uint32_t>(chIndex *
  scaleDownNonHyper), static_cast<uint32_t>(locMax));
      ++chIndex;
    }
  }

  UserAccuracyChartData nonHyperData{ std::pair(std::move(nonHyperMisc),
  std::move(nonHyperChartData)) }; UserAccuracyChartData hyperData{
  std::pair(std::move(hyperMisc), std::move(hyperChartData)) };

  return std::tuple(std::move(nonHyperData), std::move(hyperData));
  */
}

KeywordsContainer* ImageRanker::GetCorrectKwContainerPtr(DataId data_ID) const
{
  KeywordsContainer* ptr{nullptr};

  switch (std::get<0>(data_ID))
  {
    case eVocabularyId::VIRET_1200_WORDNET_2019:
      ptr = _pViretKws;
      break;

    case eVocabularyId::GOOGLE_AI_20K_2019:
      ptr = _pGoogleKws;
      break;

    default:
      LOG_ERROR("ImageRanker::GetCorrectKwContainerPtr(): Incorrect KW type!");
  }

  return ptr;
}

RelevantImagesResponse ImageRanker::GetRelevantImages(DataId data_ID,
                                                      const std::vector<std::string>& queriesEncodedPlaintext,
                                                      size_t numResults, InputDataTransformId aggId,
                                                      RankingModelId modelId, const RankingModelSettings& modelSettings,
                                                      const InputDataTransformSettings& aggSettings, size_t imageId,
                                                      bool withOccuranceValue) const
{
  std::vector<CnfFormula> formulae;

  KeywordsContainer* pKws{GetCorrectKwContainerPtr(data_ID)};

  for (auto&& queryString : queriesEncodedPlaintext)
  {
    // Decode query to logical CNF formula
    CnfFormula queryFormula{pKws->GetCanonicalQuery(EncodeAndQuery(queryString))};

    formulae.push_back(queryFormula);
  }

  // Get desired aggregation
  auto pAggFn = GetAggregationById(aggId);
  // Setup model correctly
  pAggFn->SetTransformationSettings(aggSettings);

  // Get disired model
  auto pRankingModel = GetRankingModelById(modelId);
  // Setup model correctly
  pRankingModel->SetModelSettings(modelSettings);

  // Rank it
  auto [imgOrder, targetImgRank]{pRankingModel->GetRankedImages(formulae, data_ID, pAggFn, &_indexKwFrequency,
                                                                _images, _keywordContainers, numResults, imageId)};

  RelevantImagesResponse resultResponse;

  std::vector<std::tuple<size_t, std::string, float>> occuranceHistogram;
  occuranceHistogram.reserve(pKws->GetNetVectorSize());

  if (withOccuranceValue)
  {
    //// Prefil keyword wordnetIDs
    // for (size_t i{ 0ULL }; i < _pViretKws->GetNetVectorSize(); ++i)
    //{
    //  occuranceHistogram.emplace_back(std::get<0>(_pViretKws->GetKeywordByVectorIndex(i)),
    //  std::get<1>(_pViretKws->GetKeywordByVectorIndex(i)), 0.0f);
    //}

    // for (auto&& imgId : imgOrder)
    //{
    //  auto imagePtr = GetImageDataById(imgId);
    //  auto min = imagePtr->m_min;

    //  const auto& linBinVector{ imagePtr->_transformedImageScoringData.at(200)
    //  };

    //  // Add ranking to histogram
    //  for (auto&& r : imagePtr->_rawImageScoringDataSorted)
    //  {

    //    std::get<2>(occuranceHistogram[r.first]) += linBinVector[r.first];
    //  }
    //}

    //// Sort suggested list
    // std::sort(occuranceHistogram.begin(), occuranceHistogram.end(),
    //  [](const std::tuple<size_t, std::string, float>& l, std::tuple<size_t,
    //  std::string, float>& r)
    //  {
    //    return std::get<2>(l) > std::get<2>(r);
    //  }
    //);
  }

  // Prepare final result to return
  {
    size_t i{0ULL};
    for (auto&& imgId : imgOrder)
    {
      std::get<0>(resultResponse).emplace_back(GetImageDataById(imgId));
      // std::get<1>(resultResponse).emplace_back(std::move(occuranceHistogram[i]));

      ++i;
    }
  }

  // Fill in QueryResult
  std::get<2>(resultResponse) = targetImgRank;

  return resultResponse;
}

const Image* ImageRanker::GetImageDataById(size_t imageId) const
{
  // Get correct image index
  size_t index{imageId / _imageIdStride};

  if (index >= _images.size())
  {
    LOG_ERROR("Out of bounds image index."s);
    return nullptr;
  }

  return _images[index].get();
}





// =======================================================================================
// =======================================================================================
// =======================================================================================
// =======================================================================================
// =======================================================================================
// =======================================================================================

SimulatedUser ImageRanker::GetSimUserSettings(const SimulatedUserSettings& stringSettings) const
{
  SimulatedUser newSimUser;

  // If setting 0 (Simulated user exponent) is set
  if (stringSettings.size() >= 1 && stringSettings[0].size() >= 0)
  {
    newSimUser.m_exponent = strToInt(stringSettings[0]);
  }

  return newSimUser;
}

void ImageRanker::ComputeApproxDocFrequency(size_t aggregationGuid, float treshold)
{
  LOG("Not implemented: omputeApproxDocFrequency()!");

  /*
  _indexKwFrequency.reserve(_pViretKws->GetNetVectorSize());

  std::vector<size_t> indexKwFrequencyCount;
  indexKwFrequencyCount.resize(_pViretKws->GetNetVectorSize());


  // Iterate thorough all images
  for (auto&& [id, pImg] : _images)
  {
    auto it = pImg->_transformedImageScoringData.find(aggregationGuid);

    if (it == pImg->_transformedImageScoringData.end())
    {
      LOG_ERROR("Aggregation GUID"s + std::to_string(aggregationGuid) + " not
  found.");
    }

    {
      size_t i{ 0ULL };
      for (auto&& fl : it->second)
      {
        // If this keyword is truly present
        if (fl > treshold)
        {
          // Increment it's count
          ++indexKwFrequencyCount[i];
        }

        ++i;
      }
    }
  }

  // Find maximum
  std::pair<size_t, size_t> maxIndexCount{ 0ULL, 0ULL };
  {
    size_t i{ 0ULL };
    for (auto&& indexCount : indexKwFrequencyCount)
    {
      if (indexCount > maxIndexCount.second)
      {
        maxIndexCount.first = i;
        maxIndexCount.second = indexCount;
      }

      ++i;
    }
  }

  size_t i{ 0ULL };
  for (auto&& indexCount : indexKwFrequencyCount)
  {
    _indexKwFrequency.emplace_back( logf(( (float)maxIndexCount.second /
  indexCount)));
  }
  */
}

void ImageRanker::GenerateBestHypernymsForImages()
{
  /*
  auto cmp = [](const std::pair<size_t, float>& left, const std::pair<size_t,
  float>& right)
  {
    return left.second < right.second;
  };

  for (auto&& [imgId, pImg] : _images)
  {
    std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t,
  float>>, decltype(cmp)> maxHeap(cmp);

    for (auto&& [wordnetId, pKw] : _pViretKws->_wordnetIdToKeywords)
    {

      // If has some hyponyms
      if (pKw->m_vectorIndex == ERR_VAL<size_t>())
      {
        float totalRank{ 0.0f };
        for (auto&& kwIndex : pKw->m_hyponymBinIndices)
        {
          totalRank += pImg->m_rawNetRanking[kwIndex];
        }

        maxHeap.push(std::pair(wordnetId, totalRank));
      }
    }

    while (!maxHeap.empty())
    {
      auto item = maxHeap.top();
      maxHeap.pop();

      pImg->m_hypernymsRankingSorted.emplace_back(std::move(item));
    }

  }
  */
}

void ImageRanker::InitializeGridTests()
{
  LOG("Not implemented: ImageRanker::InitializeGridTests()!");

  // ==========================================
  // Iterate through all desired configurations we want to test
  // ==========================================

  // Aggregations
  // for (auto&& agg : GridTest::m_aggregations)
  //{
  //  // Query origins
  //  for (auto&& queryOrigin : GridTest::m_queryOrigins)
  //  {
  //    // Ranking models
  //    for (auto&& model : GridTest::m_rankingModels)
  //    {

  //      switch (model)
  //      {
  //        // BooleanBucketModel
  //      case RankingModelId::cBooleanBucket:
  //
  //        // True treshold probability values
  //        for (float fi{ BooleanBucketModel::m_trueTresholdFrom }; fi <=
  //        BooleanBucketModel::m_trueTresholdTo; fi +=
  //        BooleanBucketModel::m_trueTresholdStep)
  //        {
  //          // In bucket ordering options
  //          for (auto&& qo : BooleanBucketModel::m_inBucketOrders)
  //          {
  //            std::vector<std::string> modSettings{ std::to_string(fi),
  //            std::to_string((uint8_t)qo) };

  //            GridTest::m_testSettings.emplace_back(agg, model, queryOrigin,
  //            modSettings);
  //          }
  //        }
  //        break;

  //        // BooleanMultSumMaxModel
  //      case RankingModelId::cViretBase:
  //        // True treshold probability values
  //        for (float fi{ MultSumMaxModel::m_trueTresholdFrom }; fi <=
  //        MultSumMaxModel::m_trueTresholdTo; fi += MultSumMaxModel::m_trueTresholdStep)
  //        {
  //          // Query operation options
  //          for (auto&& qo : MultSumMaxModel::m_queryOperations)
  //          {
  //            std::vector<std::string> modSettings{std::to_string(fi),
  //            std::to_string((uint8_t)qo)};

  //            GridTest::m_testSettings.emplace_back(agg, model, queryOrigin,
  //            modSettings);
  //          }
  //        }
  //        break;
  //      }

  //    }
  //  }
  //}
  LOG("GridTests initialized.");
}

std::pair<uint8_t, uint8_t> ImageRanker::GetGridTestProgress() const { return GridTest::GetGridTestProgress(); }

// \todo Export to new class Exporter
void ImageRanker::PrintIntActionsCsv() const
{
  std::string query1{
      "SELECT id, session_duration, end_status FROM "
      "`image-ranker-collector-data2`.interactive_searches;"};
  std::string query2{
      "SELECT `interactive_search_id`, `index`, `action`, `score`, `operand` "
      "FROM "
      "`image-ranker-collector-data2`.interactive_searches_actions;"};
  auto result1{_db.ResultQuery(query1)};
  auto result2{_db.ResultQuery(query2)};

  auto actionIt{result2.second.begin()};

  std::vector<std::vector<size_t>> sessProgresses;

  for (auto&& actionSess : result1.second)
  {
    std::vector<size_t> oneSess;

    size_t sessId{(size_t)strToInt(actionSess[0])};
    size_t sessDuration{(size_t)strToInt(actionSess[1])};
    size_t endStatus{(size_t)strToInt(actionSess[2])};

    bool isInitial{true};
    std::vector<std::string> initialQuery;
    std::vector<std::string> fullQuery;

    size_t actionInitialCount{0_z};
    size_t actionFinalCount{0_z};
    std::string initialRank{""};
    std::string finalRank{""};

    for (; (result2.second.end() != actionIt && strToInt((*actionIt)[0]) == sessId); ++actionIt)
    {
      auto&& actionRow = (*actionIt);

      if (actionRow[2] == "2")
      {
        isInitial = false;

        // Push rank before start of interactive refining
        oneSess.push_back(strToInt(initialRank));
      }

      if (isInitial)
      {
        ++actionInitialCount;
        initialRank = actionRow[3];
        finalRank = actionRow[3];
      }
      else
      {
        ++actionFinalCount;
        ++actionInitialCount;
        finalRank = actionRow[3];

        // Push rank
        oneSess.push_back(strToInt(finalRank));
      }

      if (actionRow[2] == "1" || actionRow[2] == "2")
      {
        if (isInitial)
        {
          initialQuery.push_back(actionRow[4]);
        }
        fullQuery.push_back(actionRow[4]);
      }
      else
      {
        for (auto it = initialQuery.begin(); it != initialQuery.end(); ++it)
        {
          if (*it == actionRow[4])
          {
            initialQuery.erase(it);
            break;
          }
        }

        for (auto it = fullQuery.begin(); it != fullQuery.end(); ++it)
        {
          if (*it == actionRow[4])
          {
            fullQuery.erase(it);
            break;
          }
        }
      }
    }

    if (initialQuery.empty() || fullQuery.empty())
    {
      continue;
    }

    std::cout << std::to_string(sessId) << "," << std::to_string(sessDuration) << "," << std::to_string(endStatus);

    {
      size_t initSize = initialQuery.size();
      size_t i{0_z};
      for (auto&& kwId : initialQuery)
      {
        std::cout << kwId;

        if (i < initSize - 1)
        {
          std::cout << "&";
        }
        ++i;
        std::cout << ",";
      }
    }
    std::cout << initialRank << ",";
    std::cout << std::to_string(actionInitialCount) << ",";

    {
      size_t initSize = fullQuery.size();
      size_t i{0_z};
      for (auto&& kwId : fullQuery)
      {
        std::cout << kwId;

        if (i < initSize - 1)
        {
          std::cout << "&";
        }
        ++i;
        std::cout << ",";
      }
    }
    if (!oneSess.empty())
    {
      sessProgresses.emplace_back(std::move(oneSess));
    }

    std::cout << finalRank << ",";
    std::cout << std::to_string(actionFinalCount) << std::endl;
  }

  std::set<size_t> m;

  for (auto&& vec : sessProgresses)
  {
    auto s{vec.size()};
    m.insert(s);
  }

  std::vector<std::vector<size_t>> ddata;

  for (auto&& size : m)
  {
    std::vector<size_t> data;
    data.resize(size, 0_z);

    size_t i{0_z};

    for (auto&& vec : sessProgresses)
    {
      if (vec.size() == size)
      {
        size_t ii{0_z};
        for (auto&& d : vec)
        {
          data[ii] += d;
          ++ii;
        }

        ++i;
      }
    }

    // Divide
    for (auto&& d : data)
    {
      d = d / i;
    }

    ddata.push_back(data);
  }
}

#endif