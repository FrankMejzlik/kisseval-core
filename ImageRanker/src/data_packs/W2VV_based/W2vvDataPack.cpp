
#include "W2vvDataPack.h"

#include <thread>

#include "./ranking_models/ranking_models.h"
#include "./transformations/transformations.h"

using namespace image_ranker;

W2vvDataPack::W2vvDataPack(const BaseImageset* p_is, const StringId& ID, const StringId& target_imageset_ID,
                           const std::string& model_options, const std::string& description, const DataPackStats& stats,
                           const W2vvDataPackRef::VocabData& vocab_data_refs,
                           std::vector<std::vector<float>>&& frame_features, Matrix<float>&& kw_features,
                           Vector<float>&& kw_bias_vec, Matrix<float>&& kw_PCA_mat, Vector<float>&& kw_PCA_mean_vec)
    : BaseDataPack(p_is, ID, target_imageset_ID, model_options, description, stats),
      _features_of_frames(std::move(frame_features)),
      _keywords(vocab_data_refs),
      _kw_features(std::move(kw_features)),
      _kw_bias_vec(std::move(kw_bias_vec)),
      _kw_PCA_mat(std::move(kw_PCA_mat)),
      _kw_PCA_mean_vec(std::move(kw_PCA_mean_vec))
{
  // Instantiate all wanted models
  _models.emplace(enum_label(eModelIds::W2VV_BOW_VBS2020).first, std::make_unique<PlainBowModel>());
}

const std::string& W2vvDataPack::get_vocab_ID() const { return _keywords.get_ID(); }

const std::string& W2vvDataPack::get_vocab_description() const { return _keywords.get_description(); }


RankingResult W2vvDataPack::rank_frames(const std::vector<CnfFormula>& user_queries, const std::string& model_commands,
                                        size_t result_size, FrameId target_image_ID) const
{
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

  // Run this model
  return ranking_model.rank_frames(_features_of_frames, _kw_features, _kw_bias_vec, _kw_PCA_mat, _kw_PCA_mean_vec,
                                   _keywords, user_queries, result_size, opt_key_vals, target_image_ID);
}

RankingResult W2vvDataPack::rank_frames(const std::vector<std::string>& user_native_queries,
                                        const std::string& model_commands, size_t result_size,
                                        FrameId target_image_ID) const
{
  std::vector<CnfFormula> cnf_queries;
  for (auto&& nat_q : user_native_queries)
  {
    cnf_queries.emplace_back(native_query_to_CNF_formula(nat_q));
  }

  return rank_frames(cnf_queries, model_commands, result_size, target_image_ID);
};

ModelTestResult W2vvDataPack::test_model(const std::vector<UserTestQuery>& test_queries,
                                         const std::string& model_commands, size_t num_points) const
{
  // Parse model & transformation
  std::vector<std::string> tokens = split(model_commands, ';');

  std::vector<ModelKeyValOption> opt_key_vals;

  std::string model_ID;
  std::string sim_user_ID;

  for (auto&& tok : tokens)
  {
    auto key_val = split(tok, '=');

    // Model ID && Transform ID
    if (key_val[0] == enum_label(eModelOptsKeys::MODEL_ID).first)
    {
      model_ID = key_val[1];
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

  // Choose desired simulated user
  auto iter_su = _sim_users.find(sim_user_ID);
  if (iter_su == _sim_users.end())
  {
    iter_su = _sim_users.find(enum_label(eSimUserIds::NO_SIM).first);
  }
  const auto& sim_user = *(iter_su->second);

  // Process sim user \todo
  // test_queries = sim_user.process_sim_user(_frames_features, _keywords, test_queries, opt_key_vals);

  return ranking_model.test_model(_features_of_frames, _kw_features, _kw_bias_vec, _kw_PCA_mat, _kw_PCA_mean_vec,
                                  _keywords, test_queries, opt_key_vals, num_points);
}

ModelTestResult W2vvDataPack::test_model(const std::vector<UserTestNativeQuery>& test_native_queries,
                                         const std::string& model_commands, size_t num_points) const
{
  std::vector<UserTestQuery> test_cnf_queries;
  test_cnf_queries.reserve(test_native_queries.size());

  for (auto&& [nat_q, target_ID] : test_native_queries)
  {
    std::vector<CnfFormula> cnf_query;
    for (auto&& single_nat_q : nat_q)
    {
      cnf_query.emplace_back(native_query_to_CNF_formula(single_nat_q));
    }
    test_cnf_queries.emplace_back(std::move(cnf_query), target_ID);
  }

  return test_model(test_cnf_queries, model_commands, num_points);
};

AutocompleteInputResult W2vvDataPack::get_autocomplete_results(const std::string& query_prefix, size_t result_size,
                                                               [[maybe_unused]] bool with_example_images,
                                                               [[maybe_unused]] const std::string& model_commands) const
{
  return { _keywords.GetNearKeywordsPtrs(query_prefix, result_size) };
}

DataPackInfo W2vvDataPack::get_info() const
{
  return DataPackInfo{ get_ID(),
                       get_description(),
                       get_model_options(),
                       target_imageset_ID(),
                       _features_of_frames.size(),
                       _keywords.get_ID(),
                       _keywords.get_description() };
}

CnfFormula W2vvDataPack::native_query_to_CNF_formula(const std::string& native_query) const
{
  std::vector<Clause> res;
  std::string nat_query(native_query);

  // Remove all unwanted charactes
  std::string illegal_chars = "\\/?!,.'\"";
  std::transform(native_query.begin(), native_query.end(), nat_query.begin(), [&illegal_chars](char c) {
    // If found in illegal, make it space
    if (illegal_chars.find(c) != std::string::npos) return ' ';

    return char(std::tolower(c));
  });

  std::stringstream query_ss(nat_query);

  // Tokenize this string
  std::string token_str;
  std::vector<std::string> query;
  size_t ignore_cnt{ 0_z };
  while (query_ss >> token_str)
  {
    auto p_kw = _keywords.GetKeywordByWord(token_str);
    if (!p_kw)
    {
      ++ignore_cnt;
      continue;
    }

    Clause c;
    c.emplace_back(Literal<KeywordId>{ p_kw->ID, false });

    res.emplace_back(std::move(c));
  }

  return res;
}
