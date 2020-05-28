

#include <json.hpp>
using json = nlohmann::json;

#include "data_packs/Google_based/GoogleVisionDataPack.h"
#include "data_packs/VIRET_based/ViretDataPack.h"
#include "data_packs/W2VV_based/W2vvDataPack.h"

#include "ImageRanker.h"

#include "datasets/SelFramesDataset.h"

#include "transformations/TransformationSoftmax.h"

using namespace image_ranker;

ImageRanker::Config ImageRanker::parse_data_config_file([[maybe_unused]] eMode mode, const std::string& filepath,
                                                        const std::string& data_dir)
{
  // Read the JSON cfg file
  std::ifstream i(filepath);
  if (!i.is_open())
  {
    std::string msg{ "Error opening file '" + filepath + "'." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
  }

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
        FrameFilenameOffsets{ is1["filename_vID_from"].get<size_t>(), is1["filename_vID_len"].get<size_t>(),
                              is1["filename_sID_from"].get<size_t>(), is1["filename_sID_len"].get<size_t>(),
                              is1["filename_fn_from"].get<size_t>(), is1["filename_fn_len"].get<size_t>() },
        data_dir + is1["frames_dir"].get<std::string>(), data_dir + is1["ID_to_frame_fpth"].get<std::string>() });
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
        dp1["ID"].get<std::string>(), dp1["type"].get<std::string>(), dp1["model_options"].get<std::string>(),
        dp1["data"]["target_dataset"].get<std::string>(),

        dp1["vocabulary"]["ID"].get<std::string>(), dp1["vocabulary"]["description"].get<std::string>(),
        data_dir + dp1["vocabulary"]["keyword_synsets_fpth"].get<std::string>(),

        data_dir + dp1["data"]["presoftmax_scorings_fpth"].get<std::string>(),
        data_dir + dp1["data"]["softmax_scorings_fpth"].get<std::string>(),
        data_dir + dp1["data"]["deep_features_fpth"].get<std::string>(), dp1["accumulated"].get<bool>() });
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
        dp3["ID"].get<std::string>(), dp3["type"].get<std::string>(), dp3["model_options"].get<std::string>(),
        dp3["data"]["target_dataset"].get<std::string>(),

        dp3["vocabulary"]["ID"].get<std::string>(), dp3["vocabulary"]["description"].get<std::string>(),
        data_dir + dp3["vocabulary"]["keyword_synsets_fpth"].get<std::string>(),

        data_dir + dp3["data"]["presoftmax_scorings_fpth"].get<std::string>() });
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
        W2vvDataPackRef{ dp3["ID"].get<std::string>(),
                         dp3["type"].get<std::string>(),
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
                         dp3["data"]["deep_features_data_offset"].get<size_t>() });
  }

  return { ImageRanker::eMode::cFullAnalytical, imagesets, VIRET_data_packs, Google_data_packs, W2VV_data_packs };
}
ImageRanker::ImageRanker(const ImageRanker::Config& cfg)
    : _settings(cfg), _fileParser(), _data_manager(this), _mode(cfg.mode)
{
  /*
   * Load all available datasets
   */
  for (auto&& pack : _settings.config.dataset_packs)
  {
    // Initialize all images
    auto frames = FileParser::parse_image_metadata(pack.imgage_to_ID_fpth, pack.offsets, 1);

    _imagesets.emplace(pack.ID, std::make_unique<SelFramesDataset>(pack.ID, pack.images_dir, std::move(frames)));
  }

  /*
   * Load all available data packs
   */
  // VIRET type
  for (auto&& pack : _settings.config.VIRET_packs)
  {
    DataPackStats stats{};

    const auto& is{ imageset(pack.target_imageset) };

    // Initialize all images
    auto [presoft_data, parse_stats] =
        FileParser::parse_VIRET_format_frame_vector_file(pack.score_data.presoftmax_scorings_fpth, is.size());
    stats.avg_num_labels_asigned = parse_stats.avg_num_labels_asigned;
    stats.median_num_labels_asigned = parse_stats.median_num_labels_asigned;

    // We don't need these yet
    auto deep_features{ Matrix<float>{} };
    auto soft_data{ Matrix<float>{} };

    _data_packs.emplace(pack.ID, std::make_unique<ViretDataPack>(&is, pack.ID, pack.target_imageset, pack.accumulated,
                                                                 pack.model_options, pack.description, stats,
                                                                 pack.vocabulary_data, std::move(presoft_data),
                                                                 std::move(soft_data), std::move(deep_features)));
  }

  // Google type
  for (auto&& pack : _settings.config.Google_packs)
  {
    DataPackStats stats{};

    const auto& is{ imageset(pack.target_imageset) };

    // Initialize all images
    // \todo Use transparent sparse matrix representation for Google data

    auto [presoft_data, parse_stats] =
        FileParser::parse_GoogleVision_format_frame_vector_file(pack.score_data.presoftmax_scorings_fpth, is.size());
    stats.avg_num_labels_asigned = parse_stats.avg_num_labels_asigned;
    stats.median_num_labels_asigned = parse_stats.median_num_labels_asigned;

    _data_packs.emplace(pack.ID, std::make_unique<GoogleVisionDataPack>(&is, pack.ID, pack.target_imageset,
                                                                        pack.model_options, pack.description, stats,
                                                                        pack.vocabulary_data, std::move(presoft_data)));
  }

  // W2VV type
  for (auto&& pack : _settings.config.W2VV_packs)
  {
    DataPackStats stats{};

    const auto& is{ imageset(pack.target_imageset) };

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
                                     &is, pack.ID, pack.target_imageset, pack.model_options, pack.description, stats,
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

  _data_manager.submit_annotator_user_queries(data_pack_ID, res->second->get_vocab_ID(),
                                              res->second->target_imageset_ID(), model_options, user_level,
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

    FrameId target_ID{ query.target_sequence_IDs.at(0) };

    // Check target validity
    if (target_ID >= is.size())
    {
      std::string msg{ "Invalid `target_frame_ID` = " + std::to_string(target_ID) + "." };
      LOGE(msg);
      PROD_THROW("Invalid parameters in function call.");
    }

    result.session_ID = query.session_ID;
    result.human_readable_query = query.user_query_readable.at(0);
    result.frame_filename = is[target_ID].m_filename;

    result.model_top_query = "";

    userResult.emplace_back(std::move(result));
  }

  return userResult;
}

const std::string& ImageRanker::get_frame_filename(const std::string& imageset_ID, size_t frame_ID) const
{
  const SelFrame& img = get_frame(imageset_ID, frame_ID);

  return img.m_filename;
}

const SelFrame& ImageRanker::get_frame(const std::string& imageset_ID, size_t frame_ID) const
{
  return imageset(imageset_ID)[frame_ID];
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
  return &(imageset(imageset_ID).random_frame());
}

AutocompleteInputResult ImageRanker::get_autocomplete_results(const std::string& data_pack_ID,
                                                              const std::string& query_prefix, size_t result_size,
                                                              bool with_example_image,
                                                              const PackModelCommands& model_options) const
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

  return dp.get_autocomplete_results(query_prefix, result_size, with_example_image, model_options);
}

std::vector<const SelFrame*> ImageRanker::frame_successors(const std::string& imageset_ID, FrameId ID,
                                                           size_t num_succs) const
{
  std::vector<const SelFrame*> res;
  res.reserve(num_succs);

  /** We see all selected frames as one whole video since it has no effect on temp query performance */
  const auto& is{ imageset(imageset_ID) };

  const SelFrame& frame = is[ID];
  res.emplace_back(&frame);
  for (size_t i{ ID + 1 }; i <= ID + num_succs && i < is.size(); ++i)
  {
    res.emplace_back(&is[i]);
  }

  return res;
}

SearchSessRankChartData ImageRanker::get_search_sessions_rank_progress_chart_data(const std::string& data_pack_ID,
                                                                                  const std::string& model_options,
                                                                                  size_t max_user_level,
                                                                                  size_t min_samples,
                                                                                  bool normalize) const
{
  const auto& dp{ data_pack(data_pack_ID) };
  const auto& is(imageset(dp.target_imageset_ID()));

  auto res{ _data_manager.get_search_sessions_rank_progress_chart_data(data_pack_ID, model_options, max_user_level,
                                                                       is.size(), min_samples, normalize) };

  return res;
}

HistogramChartData<size_t, float> ImageRanker::get_histogram_used_labels(const std::string& data_pack_ID,
                                                                         const std::string& model_options,
                                                                         size_t num_points, bool accumulated,
                                                                         [[maybe_unused]] size_t max_user_level) const
{
  constexpr eUserQueryOrigin user_query_origin{ eUserQueryOrigin::SEMI_EXPERTS };
  const auto& dp{ data_pack(data_pack_ID) };

  ModelTestResult res;
  size_t num_queries{ ERR_VAL<size_t>() };

  // Fetch queries from the DB
  auto test_queries{ _data_manager.fetch_user_test_queries(user_query_origin, dp.get_vocab_ID()) };
  num_queries = test_queries.size();

  return dp.get_histogram_used_labels(test_queries, model_options, num_queries, num_points, accumulated);
}

LoadedImagesetsInfo ImageRanker::get_loaded_imagesets_info() const
{
  std::vector<ImagesetInfo> infos;

  for (auto&& is : _imagesets)
  {
    infos.emplace_back(is.second->get_info());
  }

  return LoadedImagesetsInfo{ infos };
}

LoadedDataPacksInfo ImageRanker::get_loaded_data_packs_info() const
{
  std::vector<DataPackInfo> infos;

  for (auto&& dp : _data_packs)
  {
    infos.emplace_back(dp.second->get_info());
  }

  return LoadedDataPacksInfo{ infos };
}

RankingResultWithFilenames ImageRanker::rank_frames(const std::vector<std::string>& user_queries,
                                                    const DataPackId& data_pack_ID,
                                                    const PackModelCommands& model_commands, size_t result_size,
                                                    bool native_lang_queries, FrameId target_image_ID) const
{
  const BaseDataPack& dp{ data_pack(data_pack_ID) };
  const BaseImageset& is{ imageset(dp.target_imageset_ID()) };

  // Decide if native or ID based version wanted
  RankingResult res{};
  if (native_lang_queries)
  {
    res = dp.rank_frames(user_queries, model_commands, result_size, target_image_ID);
  }
  else
  {
    // Parse CNF strings
    std::vector<CnfFormula> cnf_user_query;
    cnf_user_query.reserve(user_queries.size());
    for (auto&& single_query : user_queries)
    {
      cnf_user_query.emplace_back(parse_cnf_string(single_query));
    }

    res = dp.rank_frames(cnf_user_query, model_commands, result_size, target_image_ID);
  }

  RankingResultWithFilenames res_with_filenames;
  res_with_filenames.target = res.target;
  res_with_filenames.target_pos = res.target_pos;

  std::transform(res.m_frames.begin(), res.m_frames.end(), std::back_inserter(res_with_filenames.m_frames),
                 [&is](const FrameId& f_ID) {
                   auto filename{ is[f_ID].m_filename };
                   return std::pair<FrameId, std::string>(f_ID, filename);
                 });

  return res_with_filenames;
}

ModelTestResult ImageRanker::run_model_test(eUserQueryOrigin queries_origin, const DataPackId& data_pack_ID,
                                            const PackModelCommands& model_options, bool native_lang_queries,
                                            size_t num_points, bool normalize_y) const
{
  const auto& dp{ data_pack(data_pack_ID) };

  ModelTestResult res;
  size_t num_queries{ ERR_VAL<size_t>() };

  // Decide if native or ID based version wanted
  if (native_lang_queries)
  {
    // Fetch queries from the DB
    auto native_test_queries{ _data_manager.fetch_user_native_test_queries(queries_origin, dp.target_imageset_ID()) };
    num_queries = native_test_queries.size();
    std::cout << "num " << num_queries << std::endl;

    res = dp.test_model(native_test_queries, model_options, num_points);
  }
  else
  {
    // Fetch queries from the DB
    auto test_queries{ _data_manager.fetch_user_test_queries(queries_origin, dp.get_vocab_ID(), data_pack_ID) };
    num_queries = test_queries.size();

    res = dp.test_model(test_queries, model_options, num_points);
  }

  if (normalize_y)
  {
    std::transform(res.begin(), res.end(), res.begin(), [num_queries](const std::pair<uint32_t, uint32_t>& x) {
      return std::pair(x.first, uint32_t((float(x.second) / num_queries) * 10000.0F));  // NOLINT
    });
  }

  return res;
}

void ImageRanker::submit_search_session(const std::string& data_pack_ID, const std::string& model_options,
                                        size_t user_level, bool with_example_images, FrameId target_frame_ID,
                                        eSearchSessionEndStatus end_status, size_t duration,
                                        const std::string& sessionId,
                                        const std::vector<InteractiveSearchAction>& actions)
{
  const BaseDataPack& dp{ data_pack(data_pack_ID) };
  const std::string& imageset_ID{ dp.get_vocab_ID() };

  _data_manager.submit_search_session(data_pack_ID, imageset_ID, model_options, user_level, with_example_images,
                                      target_frame_ID, end_status, duration, sessionId, actions);
}

FrameDetailData ImageRanker::get_frame_detail_data(FrameId frame_ID, const std::string& data_pack_ID,
                                                   const std::string& model_commands,
                                                   [[maybe_unused]] bool with_example_frames, bool accumulated)
{
  const BaseDataPack& dp{ data_pack(data_pack_ID) };

  // Parse model & transformation
  std::vector<std::string> tokens = split(model_commands, ';');

  std::vector<ModelKeyValOption> opt_key_vals;

  for (auto&& tok : tokens)
  {
    auto key_val = split(tok, '=');
    opt_key_vals.emplace_back(key_val[0], key_val[1]);
  }

  // Get top keywords for the given frame ID
  std::vector<const Keyword*> top_keywords{ dp.get_frame_top_classes(frame_ID, opt_key_vals, accumulated) };

  // Cache it up if needed
  dp.cache_up_example_images(top_keywords, model_commands);

  FrameDetailData res_data;
  res_data.frame_ID = frame_ID;
  res_data.data_pack_ID = data_pack_ID;
  res_data.model_options = model_commands;
  res_data.top_keywords = std::move(top_keywords);

  return res_data;
}
