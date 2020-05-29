
#include "ViretDataPack.h"

#include <algorithm>
#include <thread>

#include "./datasets/BaseImageset.h"
#include "./ranking_models/ranking_models.h"
#include "./transformations/transformations.h"

using namespace image_ranker;

ViretDataPack::ViretDataPack(const BaseImageset* p_is, const StringId& ID, const StringId& target_imageset_ID,
                             bool accumulated, const std::string& model_options, const std::string& description,
                             const DataPackStats& stats, const ViretDataPackRef::VocabData& vocab_data_refs,
                             std::vector<std::vector<float>>&& presoft, std::vector<std::vector<float>>&& softmax_data,
                             std::vector<std::vector<float>>&& feas_data)
    : BaseDataPack(p_is, ID, target_imageset_ID, model_options, description, stats),
      _keywords(vocab_data_refs),
      _deep_feas_data_raw(std::move(feas_data)),
      _presoftmax_data_raw(std::move(presoft)),
      _softmax_data_raw(std::move(softmax_data))
{
  // Instantiate all wanted transforms
  std::thread t1([this, accumulated]() {
    _transforms.emplace(enum_label(eTransformationIds::SOFTMAX).first,
                        std::make_unique<TransformationSoftmax>(_keywords, _presoftmax_data_raw, !accumulated));
  });
  std::thread t2([this, accumulated]() {
    _transforms.emplace(enum_label(eTransformationIds::LINEAR_01).first,
                        std::make_unique<TransformationLinear01>(_keywords, _presoftmax_data_raw, !accumulated));
  });

  // Instantiate all wanted models
  _models.emplace(enum_label(eModelIds::BOOLEAN).first, std::make_unique<BooleanModel>());
  _models.emplace(enum_label(eModelIds::VECTOR_SPACE).first, std::make_unique<VectorSpaceModel>());
  _models.emplace(enum_label(eModelIds::MULT_SUM_MAX).first, std::make_unique<MultSumMaxModel>());

  // Simulated user models
  _sim_users.emplace(enum_label(eSimUserIds::NO_SIM).first, std::make_unique<SimUserNoSim>());
  _sim_users.emplace(enum_label(eSimUserIds::USER_X_TO_P).first, std::make_unique<SimUserXToP>());

  t1.join();
  t2.join();
}

HistogramChartData<size_t, float> ViretDataPack::get_histogram_used_labels(
    const std::vector<UserTestQuery>& test_queries, const std::string& model_commands,
    [[maybe_unused]] size_t num_queries, size_t num_points, bool accumulated) const
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

  std::string transform_ID{ "linear_01" };

  for (auto&& tok : tokens)
  {
    auto key_val = split(tok, '=');
    if (key_val[0] == enum_label(eModelOptsKeys::TRANSFORM_ID).first)
    {
      transform_ID = key_val[1];
    }
    // Options for model itself
    else
    {
      opt_key_vals.emplace_back(key_val[0], key_val[1]);
    }
  }

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOGE("Uknown transform_ID: '" + transform_ID + "'.");
  }
  const auto& transform = *(iter_t->second);

  const std::vector<std::vector<float>>& data_mat{ accumulated ? transform.data_sum() : transform.data_linear_raw() };

  // Convert this raw matrix to sorted matrix of pairs with their indices
  using IdxScorePair = std::pair<size_t, float>;
  std::vector<std::vector<IdxScorePair>> tagged_sorted_mat;
  tagged_sorted_mat.resize(data_mat.size());

  std::transform(
      data_mat.cbegin(), data_mat.cend(), tagged_sorted_mat.begin(),
      [](const std::vector<float>& v) {
        // Ptr to the first elem for index calculation
        auto base{ &v.front() };

        // Tag each element with the according index
        std::vector<std::pair<size_t, float>> tagged_sorted_vec;
        tagged_sorted_vec.resize(v.size());
        std::transform(v.cbegin(), v.cend(), tagged_sorted_vec.begin(), [base](const float& v) {
          size_t idx{ size_t(&v - base) };
          return std::pair(idx, v);
        });

        // Sort this vector
        std::sort(tagged_sorted_vec.begin(), tagged_sorted_vec.end(),
                  [](const std::pair<size_t, float>& l, std::pair<size_t, float>& r) { return l.second > r.second; });
        return tagged_sorted_vec;
      });

  std::vector<size_t> label_hits;
  label_hits.resize(tagged_sorted_mat.front().size());

  // Iterate over all user queries
  size_t hit_count{ 0_z };
  for (auto&& [user_queries, target_frame_ID] : idx_test_queries)
  {
    const auto& frame_vec{ tagged_sorted_mat[target_frame_ID] };

    // Find this labels
    for (auto&& c : user_queries.front())
    {
      for (auto&& lit : c)
      {
        KeywordId kw_ID{ lit.atom };

        size_t i = 0;
        for (auto&& [idx, score] : frame_vec)
        {
          if (idx == kw_ID)
          {
            break;
          }
          ++i;
        }

        assert(i < frame_vec.size());

        // Add hit
        ++hit_count;
        ++label_hits[i];
      }
    }
  }

  float div{ tagged_sorted_mat.front().size() / float(num_points) };
  std::vector<size_t> scaled_label_hits;
  scaled_label_hits.resize(num_points);

  {
    size_t ii{ 0_z };
    for (auto&& hits : label_hits)
    {
      size_t scaled_idx{ size_t(ii / div) };

      // \todo What function here?
      scaled_label_hits[scaled_idx] = std::max(scaled_label_hits[scaled_idx], hits);
      // AVG?
      // scaled_label_hits[scaled_idx] +=  hits;

      ++ii;
    }
  }

  std::vector<size_t> normalized_label_xs;
  normalized_label_xs.reserve(scaled_label_hits.size());

  size_t sum_scaled{ 0_z };
  {
    size_t ii{ 0_z };
    for (auto&& val : scaled_label_hits)
    {
      sum_scaled += val;

      normalized_label_xs.emplace_back(size_t(div * ii));
      ++ii;
    }
  }

  // normalize

  std::vector<float> normalized_label_hits;
  normalized_label_hits.reserve(scaled_label_hits.size());
  std::transform(scaled_label_hits.begin(), scaled_label_hits.end(), std::back_inserter(normalized_label_hits),
                 [sum_scaled](const size_t& val) { return float(val) / sum_scaled; });

  HistogramChartData<size_t, float> res;

  res.x = normalized_label_xs;
  res.fx = normalized_label_hits;
  return res;
}

const std::string& ViretDataPack::get_vocab_ID() const { return _keywords.get_ID(); }

const std::string& ViretDataPack::get_vocab_description() const { return _keywords.get_description(); }

RankingResult ViretDataPack::rank_frames(const std::vector<CnfFormula>& user_queries, const std::string& model_commands,
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
    PROD_THROW("Data error.")
  }
  const auto& ranking_model = *(iter_m->second);

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOGE("Uknown transform_ID: '" + transform_ID + "'.");
    PROD_THROW("Data error.")
  }
  const auto& transform = *(iter_t->second);

  // Run this model
  return ranking_model.rank_frames(transform, _keywords, idx_queries, result_size, opt_key_vals, target_image_ID);
}

ModelTestResult ViretDataPack::test_model(const std::vector<UserTestQuery>& test_queries,
                                          const std::string& model_commands, size_t num_points) const
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
    if (!_models.empty())
    {
      iter_m = _models.begin();
      LOGW("Uknown model_ID: '" + model_ID + "'. Falling back to: " + iter_m->first);
    }
    else
    {
      LOGE("Uknown models!");
      PROD_THROW("Data error.")
    }
  }
  const auto& ranking_model = *(iter_m->second);

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    if (!_transforms.empty())
    {
      iter_t = _transforms.begin();
      LOGW("Uknown transform_ID: '" + transform_ID + "'. Falling back to: " + iter_t->first);
    }
    else
    {
      LOGE("No transformations!");
      PROD_THROW("Data error.")
    }
  }
  const auto& transform = *(iter_t->second);

  // Choose desired simulated user
  auto iter_su = _sim_users.find(sim_user_ID);
  if (iter_su == _sim_users.end())
  {
    if (!_sim_users.empty())
    {
      iter_su = _sim_users.begin();
    }
    else
    {
      LOGE("No sim users!");
      PROD_THROW("Data error.")
    }
  }
  const auto& sim_user = *(iter_su->second);

  // Process sim user
  // Use linear transform data without hypernyms accumulated
  const auto& sim_user_tr{ *(_transforms.find(enum_label(eTransformationIds::LINEAR_01).first)->second) };
  idx_test_queries =
      sim_user.process_sim_user(get_imageset_ptr(), sim_user_tr, _keywords, idx_test_queries, opt_key_vals);

  return ranking_model.test_model(transform, _keywords, idx_test_queries, opt_key_vals, num_points);
}

AutocompleteInputResult ViretDataPack::get_autocomplete_results(const std::string& query_prefix, size_t result_size,
                                                                [[maybe_unused]] bool with_example_images,
                                                                const std::string& model_commands) const
{
  auto kws = const_cast<const KeywordsContainer&>(_keywords).get_near_keywords(query_prefix, result_size); // NOLINT

  // Cache it up if needed
  cache_up_example_images(kws, model_commands);

  AutocompleteInputResult res{ kws };

  return res;
}

std::vector<const Keyword*> ViretDataPack::get_frame_top_classes(FrameId frame_ID,
                                                                 const std::vector<ModelKeyValOption>& opt_key_vals,
                                                                 bool accumulated) const
{
  bool max_based{ false };
  std::string transform_ID{ "linear_01" };

  for (auto&& [key, val] : opt_key_vals)
  {
    if (key == "transform")
    {
      transform_ID = val;
    }
    else if (key == "model_operations")
    {
      // If max-based model
      if (val == "mult-max" || val == "sum-max" || val == "max-max")
      {
        max_based = true;
      }
    }
  }

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOGE("Tranform not found!");
  }
  const auto& transform = *(iter_t->second);

  const DataInfo* p_data_info{ nullptr };

  if (accumulated)
  {
    p_data_info = max_based ? &transform.data_max_info() : &transform.data_sum_info();
  }
  else
  {
    p_data_info = &transform.data_linear_raw_info();
  }

  const auto& kw_IDs{ p_data_info->top_classes[frame_ID] };

  std::vector<const Keyword*> res;

  std::transform(kw_IDs.begin(), kw_IDs.end(), std::back_inserter(res),
                 [this](const size_t& kw_ID) { return &_keywords[kw_ID]; });

  return res;
};

void ViretDataPack::cache_up_example_images(const std::vector<const Keyword*>& kws,
                                            const std::string& model_commands) const
{
  auto hasher{ std::hash<std::string>{} };

  size_t curr_opts_hash{ hasher(model_commands) };

  for (auto&& cp_kw : kws)
  {
    const auto& all_kws_with_ID{ _keywords.get_all_keywords_ptrs(cp_kw->ID) };

    // Check if images are already cached
    Keyword& kw{ **all_kws_with_ID.begin() };
    if (curr_opts_hash == kw.last_examples_hash)
    {
      continue;
    }

    // Get example images
    CnfFormula fml{ Clause{ Literal<KeywordId>{ kw.ID } } };
    std::vector<CnfFormula> v{ fml };

    // Rank frames with query "this_kw"
    auto ranked_frames{ rank_frames(v, model_commands, NUM_EXAMPLE_FRAMES) };

    // Store them in all keyword synonyms
    for (auto&& p_kw : all_kws_with_ID)
    {
      p_kw->example_frames_filenames.clear();
      for (auto&& f_ID : ranked_frames.m_frames)
      {
        std::string filename{ get_imageset_ptr()->operator[](f_ID).m_filename };
        p_kw->example_frames_filenames.emplace_back(std::move(filename));
      }

      // Update hash
      p_kw->last_examples_hash = curr_opts_hash;
    }
  }
}

DataPackInfo ViretDataPack::get_info() const
{
  return DataPackInfo{ get_ID(),
                       get_description(),
                       get_model_options(),
                       target_imageset_ID(),
                       _presoftmax_data_raw.size(),
                       _keywords.get_ID(),
                       _keywords.get_description() };
}

CnfFormula ViretDataPack::keyword_IDs_to_vector_indices(CnfFormula ID_query) const
{
  for (auto&& ID_clause : ID_query)
  {
    assert(ID_clause.size() == 1);

    KeywordId kw_ID{ ID_clause[0].atom };

    const Keyword& kw{ _keywords[kw_ID] };

    std::unordered_set<size_t> vecIds;
    _keywords.get_keyword_hyponyms_indices_set_nearest(vecIds, kw.wordnet_ID);

    ID_clause.clear();
    for (auto&& id : vecIds)
    {
      ID_clause.emplace_back(Literal<KeywordId>{ id, false });
    }
  }
  return ID_query;
}

CnfFormula ViretDataPack::wordnet_IDs_to_vector_indices(CnfFormula ID_query) const
{
  for (auto&& ID_clause : ID_query)
  {
    assert(ID_clause.size() == 1);

    KeywordId kw_ID{ ID_clause[0].atom };

    std::unordered_set<size_t> vecIds;
    _keywords.get_keyword_hyponyms_indices_set_nearest(vecIds, kw_ID);

    ID_clause.clear();
    for (auto&& id : vecIds)
    {
      ID_clause.emplace_back(Literal<KeywordId>{ id, false });
    }
  }
  return ID_query;
}
