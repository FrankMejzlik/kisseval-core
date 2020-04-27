
#include "W2vvDataPack.h"

#include <thread>

#include "./ranking_models/ranking_models.h"
#include "./transformations/transformations.h"

using namespace image_ranker;

W2vvDataPack::W2vvDataPack(const StringId& ID, const StringId& target_imageset_ID, const std::string& model_options,
                           const std::string& description, const W2vvDataPackRef::VocabData& vocab_data_refs,
                           std::vector<std::vector<float>>&& presoft)
    : BaseDataPack(ID, target_imageset_ID, model_options, description),
      _presoftmax_data_raw(std::move(presoft)),
      _keywords(vocab_data_refs)
{
}

const std::string& W2vvDataPack::get_vocab_ID() const { return _keywords.get_ID(); }

const std::string& W2vvDataPack::get_vocab_description() const { return _keywords.get_description(); }

std::string W2vvDataPack::humanize_and_query(const std::string& and_query) const 
{
  LOGW("Not implemented!");
  return ""s;
}

std::vector<Keyword*> W2vvDataPack::top_frame_keywords(FrameId frame_ID) const
{
  LOGW("Not implemented!");

  return std::vector({
      _keywords.GetKeywordPtrByVectorIndex(0),
      _keywords.GetKeywordPtrByVectorIndex(1),
      _keywords.GetKeywordPtrByVectorIndex(2),
  });
}

RankingResult W2vvDataPack::rank_frames(const std::vector<CnfFormula>& user_queries, PackModelCommands model_commands,
                                        size_t result_size, FrameId target_image_ID) const
{
  // Expand query to vector indices
  std::vector<CnfFormula> idx_queries;
  idx_queries.reserve(user_queries.size());
  for (auto&& q : user_queries)
  {
    idx_queries.emplace_back(keyword_IDs_to_vector_indices(q));
  }

  std::vector<std::string> tokens = split(model_commands, ';');

  std::vector<ModelKeyValOption> opt_key_vals;

  std::string model_ID;
  std::string transform_ID;

  for (auto&& tok : tokens)
  {
    auto key_val = split(tok, '=');

    // Model ID && Transform ID
    if (key_val[0] == enum_label(eModelOptsKeys::MODEL_ID).first)
    {
      model_ID = key_val[1];
    }
    else if (key_val[0] == enum_label(eModelOptsKeys::TRANSFORM_ID).first)
    {
      transform_ID = key_val[1];
    }
    // Options for model itself
    else
    {
      opt_key_vals.emplace_back(key_val[0], key_val[1]);
    }
  }

  // Choose desired model
  auto iter_m = _models.find(model_ID);
  if (iter_m == _models.end())
  {
    LOGE("Uknown model_ID: '" + model_ID + "'.");
    return RankingResult{};
  }
  const auto& ranking_model = *(iter_m->second);

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOGE("Uknown transform_ID: '" + transform_ID + "'.");
    return RankingResult{};
  }
  const auto& transform = *(iter_t->second);

  // Run this model
  return ranking_model.rank_frames(transform, _keywords, idx_queries, result_size, opt_key_vals, target_image_ID);
}

#if 0
void do_test20()
{
#define IMG_FEATURES_20K \
  R"(c:\Users\Frank Mejzlik\data\imageset2_V3C1_VBS2020\subset_20\vis_vecs_bow-20000x2048floats.bin)"

#define EMBEDDED_QUERY_VECS_20K \
  R"(c:\Users\Frank Mejzlik\data\imageset2_V3C1_VBS2020\subset_20\txt_vecs_bow-202x2048floats.bin)"

#define TEST_FILEPATH_20 \
  R"(c:\Users\Frank Mejzlik\data\imageset2_V3C1_VBS2020\subset_20\user_annotator_queries.native.csv)"

#if DO_PCA

  auto img_features = Parser::parse_float_matrix(PATH_IMG_FEATURES_W2VV, 128, 12);
#else
  auto img_features = Parser::parse_float_matrix(IMG_FEATURES_20K, 2048, 0);
#endif

  auto embedded_sent_vecs = Parser::parse_float_matrix(EMBEDDED_QUERY_VECS_20K, 2048, 0);

  std::cout << "====================================================" << std::endl;
  std::cout << "=========== EMBEDDED BY TOMAS 20K : ==============" << std::endl;
  for (auto&& vec : embedded_sent_vecs)
  {
    for (auto&& val : vec)
    {
      std::cout << val << "\t ";
    }
    std::cout << std::endl;
    break;
  }
  std::cout << "====================================================" << std::endl;

  auto kw_features = Parser::parse_float_matrix(KW_SCORES_MAT_FILE, 2048, 0);
  auto dictionary = Parser::parse_w2vv_word_to_idx_file(W2VV_WORD_TO_IDX_FILEPATH);
  auto bias_vec_transposed = Parser::parse_float_vector(KW_BIAS_VEC_FILE, 2048, 0);

  auto PCA_mean_vec_transposed = Parser::parse_float_vector(KW_PCA_MEAN_VEC_FILE, 2048, 0);
  auto PCA_mat = Parser::parse_float_matrix(KW_PCA_MAT_FILE, 2048, 0);

  KwRanker ranker(std::move(img_features), std::move(kw_features), std::move(dictionary),
                  std::move(bias_vec_transposed), std::move(PCA_mat), std::move(PCA_mean_vec_transposed));

  constexpr uint32_t num_imgs = 20000;

  std::ifstream if_test_queries;
  if_test_queries.open(TEST_FILEPATH_20);

  if (!if_test_queries.is_open()) throw std::runtime_error("error openin file");

  std::vector<std::pair<uint32_t, std::vector<std::string>>> test_queries;

  std::string line;
  while (std::getline(if_test_queries, line))
  {
    std::stringstream line_ss(line);

    std::string token1;
    std::getline(line_ss, token1, ',');
    std::string token2;
    std::getline(line_ss, token2, ',');

    std::string query_str;
    std::getline(line_ss, query_str);

    std::stringstream id_ss(token1);
    uint32_t target_ID;
    id_ss >> target_ID;

    // Make first and last quotes space
    query_str[0] = ' ';
    query_str[query_str.length() - 1] = ' ';

    // Remove all unwanted charactes
    std::string illegal_chars = "\\/?!,.'\"";
    std::transform(query_str.begin(), query_str.end(), query_str.begin(), [&illegal_chars](char c) {
      // If found in illegal, make it space
      if (illegal_chars.find(c) != std::string::npos) return ' ';

      return c;
    });

    std::stringstream query_ss(query_str);

    std::string token_str;
    std::vector<std::string> query;
    while (query_ss >> token_str)
    {
      query.emplace_back(token_str);
    }

    test_queries.emplace_back(target_ID, std::move(query));
  }
  if_test_queries.close();

  uint32_t num_points = 100;
  uint32_t imgs_per_point = num_imgs / num_points;

  std::vector<uint32_t> chart_data;
  chart_data.resize(num_points + 1);

  uint32_t sum_rank = 0;

  uint32_t total_miss_kws = 0;
  uint32_t progress_i = 0;
  uint32_t ijj = 0;
  for (auto&& [target_ID, query] : test_queries)
  {
    // auto res = ranker.rank_query(query, num_imgs, total_miss_kws);

    auto res = ranker.rank_query_no_embed(query, num_imgs, total_miss_kws, embedded_sent_vecs[ijj]);
    ++ijj;

    uint32_t i_rank = 0;
    for (auto&& img_ID : res)
    {
#if REDUCED_DATASET
      if ((img_ID * REDUCTION_COEF) == target_ID)
#else
      if (img_ID == target_ID)
#endif
      {
        uint32_t rank = i_rank * REDUCTION_COEF;

        uint32_t rank_scaled = rank / (imgs_per_point * REDUCTION_COEF);

        ++chart_data[rank_scaled];

        sum_rank += rank;

        break;
      }

      ++i_rank;
    }
    ++progress_i;
    {
      std::cout << ">> " << progress_i << "/" << test_queries.size() << std::endl;
    }
  }

  // Accumuate results
  uint32_t accum = 0;
  uint32_t i = 0;
  for (auto&& val : chart_data)
  {
    std::cout << i * (imgs_per_point * REDUCTION_COEF) << "," << accum << std::endl;
    accum += val;
    ++i;
  }
  std::cout << num_imgs << "," << accum << std::endl;

  float avg_rank = float(sum_rank) / test_queries.size();
  float avg_miss = float(total_miss_kws) / test_queries.size();

  std::cout << "++++++++++++++++++++" << std::endl;
  std::cout << "avg_rank = " << avg_rank << std::endl;
  std::cout << "avg_kw_miss = " << avg_miss << std::endl;
  std::cout << "++++++++++++++++++++" << std::endl;
  std::cout << "++++++++++++++++++++" << std::endl;
}

#endif

ModelTestResult W2vvDataPack::test_model(const std::vector<UserTestQuery>& test_queries,
                                         PackModelCommands model_commands, size_t num_points) const
{
  // Expand query to vector indices
  std::vector<UserTestQuery> idx_test_queries;
  idx_test_queries.reserve(test_queries.size());

  for (auto&& [test_query, target_ID] : test_queries)
  {
    std::vector<CnfFormula> idx_query;
    idx_query.reserve(test_query.size());
    for (auto&& q : test_query)
    {
      idx_query.emplace_back(keyword_IDs_to_vector_indices(q));
    }
    idx_test_queries.emplace_back(std::move(idx_query), target_ID);
  }

  // Parse model & transformation
  std::vector<std::string> tokens = split(model_commands, ';');

  std::vector<ModelKeyValOption> opt_key_vals;

  std::string model_ID;
  std::string transform_ID;
  std::string sim_user_ID;

  for (auto&& tok : tokens)
  {
    auto key_val = split(tok, '=');

    // Model ID && Transform ID
    if (key_val[0] == enum_label(eModelOptsKeys::MODEL_ID).first)
    {
      model_ID = key_val[1];
    }
    else if (key_val[0] == enum_label(eModelOptsKeys::TRANSFORM_ID).first)
    {
      transform_ID = key_val[1];
    }
    else if (key_val[0] == enum_label(eModelOptsKeys::SIM_USER_ID).first)
    {
      sim_user_ID = key_val[1];
    }
    // Options for model itself
    else
    {
      opt_key_vals.emplace_back(key_val[0], key_val[1]);
    }
  }

  // Choose desired model
  auto iter_m = _models.find(model_ID);
  if (iter_m == _models.end())
  {
    LOGE("Uknown model_ID: '" + model_ID + "'.");
    return ModelTestResult{};
  }
  const auto& ranking_model = *(iter_m->second);

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOGE("Uknown transform_ID: '" + transform_ID + "'.");
    return ModelTestResult{};
  }
  const auto& transform = *(iter_t->second);

  // Choose desired simulated user
  auto iter_su = _sim_users.find(sim_user_ID);
  if (iter_su == _sim_users.end())
  {
    LOGW("sim_user_ID not found: '" + sim_user_ID + "'. Using default: " + enum_label(eSimUserIds::NO_SIM).first);

    iter_su = _sim_users.find(enum_label(eSimUserIds::NO_SIM).first);
  }
  const auto& sim_user = *(iter_su->second);

  // Process sim user
  idx_test_queries = sim_user.process_sim_user(transform, _keywords, idx_test_queries, opt_key_vals);

  return ranking_model.test_model(transform, _keywords, idx_test_queries, opt_key_vals, num_points);
}

AutocompleteInputResult W2vvDataPack::get_autocomplete_results(const std::string& query_prefix, size_t result_size,
                                                               bool with_example_image) const
{
  return {_keywords.GetNearKeywordsPtrs(query_prefix, result_size)};
}

DataPackInfo W2vvDataPack::get_info() const
{
  return DataPackInfo{get_ID(),           get_description(),          get_model_options(), target_imageset_ID(),
                      _keywords.get_ID(), _keywords.get_description()};
}

CnfFormula W2vvDataPack::keyword_IDs_to_vector_indices(CnfFormula ID_query) const
{
  // Google vocabulary has no hypernyms
  return ID_query;
}
