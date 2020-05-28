#pragma once

#include <string>
using namespace std::literals;

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <json.hpp>
using json = nlohmann::json;

#include "common.h"

using namespace image_ranker;

template <typename F>
struct scope_exit
{
  F f_;
  bool run_;
  explicit scope_exit(F f) noexcept : f_(std::move(f)), run_(true) {}
  scope_exit(scope_exit&& rhs) noexcept : f_((rhs.run_ = false, std::move(rhs.f_))), run_(true) {}
  ~scope_exit()
  {
    if (run_) f_();  // RAII semantics apply, expected not to throw
  }

  // "in place" construction expected, no default ctor provided either
  // also unclear what should be done with the old functor, should it
  // be called since it is no longer needed, or not since *this is not
  // going out of scope just yet...
  scope_exit& operator=(scope_exit&& rhs) = delete;
  // to be explicit...
  scope_exit(scope_exit const&) = delete;
  scope_exit& operator=(scope_exit const&) = delete;
};

template <typename F>
scope_exit<F> make_scope_exit(F&& f) noexcept
{
  return scope_exit<F>{ std::forward<F>(f) };
}

template <typename T>
inline T rand(T from, T to)
{
  std::uniform_int_distribution<T> uniform_dist{ from, to };
  std::random_device rd;
  std::mt19937 engine(rd());
  std::function<T()> rand = std::bind(uniform_dist, engine);

  return rand();
}

inline bool is_arch_LE()
{
  if constexpr (sizeof(char) >= sizeof(uint32_t))
  {
    throw std::runtime_error(
        "This test assumes char type "
        "smaller than uint32_t.");
  }

  uint32_t num{ 1 };
  return *(reinterpret_cast<char*>(&num)) == 1;  // NOLINT
}

/**
 * Computes outer sum of discrete function defined by given chart data
 */
inline float calc_chart_area(const ModelTestResult& chart_data)
{
  float area{ 0.0F };
  uint32_t prev_x{ 0 };
  for (auto&& [x, fx] : chart_data)
  {
    area += float((x - prev_x) * fx);

    prev_x = x;
  }

  return area;
}

/**
 * Parses string representation of tree CNF formula.
 *
 * EXAMPLE INPUT: &-20+--1+-3++-55+-333+
 */
inline CnfFormula parse_cnf_string(const std::string& string)
{
  std::vector<Clause> result;

  std::stringstream idx_ss;
  KeywordId idx_buffer;
  bool negate_next_atom{ false };

  size_t depth{ 0 };

  Clause clause_buffer;

  for (auto&& c : string)
  {
    if (bool(std::isdigit(int(c))))
    {
      idx_ss << c;
      continue;
    }

    // Flush index SS
    if (idx_ss.rdbuf()->in_avail() > 0)
    {
      idx_ss >> idx_buffer;
      clause_buffer.emplace_back(Literal<KeywordId>{ idx_buffer, negate_next_atom });
      idx_ss = std::stringstream();
    }

    if (c == '&' || c == '|')
    {
      continue;
    }

    if (c == '-')
    {
      ++depth;
      continue;
    }

    if (c == '+')
    {
      --depth;
      if (depth == 0)
      {
        // Dispatch clause
        result.emplace_back(clause_buffer);
        clause_buffer = Clause();
      }
      continue;
    }

    if (c == '~')
    {
      negate_next_atom = true;
      continue;
    }
  }

  return result;
}

inline std::vector<std::string> split(const std::string& str, char delim)
{
  std::vector<std::string> result;
  std::stringstream ss(str);
  std::string item;

  while (getline(ss, item, delim))
  {
    result.emplace_back(item);
  }

  return result;
}

inline std::vector<std::string> split(const std::string& str, const std::string& delim)
{
  std::vector<std::string> result;

  std::string s{ str };

  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delim)) != std::string::npos)
  {
    token = s.substr(0, pos);
    result.emplace_back(token);
    s.erase(0, pos + delim.length());
  }
  result.emplace_back(s);

  return result;
}

inline std::array<char, 4> floatToBytesLE(float number)
{
  char* bit_number{ reinterpret_cast<char*>(&number) };  // NOLINT

  std::array<char, 4> buff{ bit_number[0], bit_number[1], bit_number[2], bit_number[3] };  // NOLINT

  if (!is_arch_LE())
  {
    buff[0] = bit_number[3];  // NOLINT
    buff[1] = bit_number[2];  // NOLINT
    buff[2] = bit_number[1];  // NOLINT
    buff[3] = bit_number[0];  // NOLINT
  }

  return buff;
}

inline std::array<char, 4> uint32ToBytesLE(uint32_t number)
{
  char* bit_number{ reinterpret_cast<char*>(&number) };  // NOLINT

  std::array<char, 4> buff{ bit_number[0], bit_number[1], bit_number[2], bit_number[3] };  // NOLINT

  if (!is_arch_LE())
  {
    buff[0] = bit_number[3];  // NOLINT
    buff[1] = bit_number[2];  // NOLINT
    buff[2] = bit_number[1];  // NOLINT
    buff[3] = bit_number[0];  // NOLINT
  }

  return buff;
}

inline int32_t ParseIntegerLE(const char* buff)
{
  int32_t sign_num{ *(reinterpret_cast<const int32_t*>(buff)) };  // NOLINT

  // If BE
  if (!is_arch_LE())
  {
    sign_num =
        static_cast<int32_t>(static_cast<uint32_t>(buff[0]) << 24 | static_cast<uint32_t>(buff[1]) << 16 |  // NOLINT
                             static_cast<uint32_t>(buff[2]) << 8 | static_cast<uint32_t>(buff[3]));         // NOLINT
  }

  // Return parsed integer
  return sign_num;
}

inline int32_t ParseIntegerLE(const std::vector<char>& buff)
{
  int32_t sign_num{ *(reinterpret_cast<const int32_t*>(buff.data())) };  // NOLINT

  // Construct final BE integer
  if (!is_arch_LE())
  {
    sign_num =
        static_cast<int32_t>(static_cast<uint32_t>(buff[0]) << 24 | static_cast<uint32_t>(buff[1]) << 16 |  // NOLINT
                             static_cast<uint32_t>(buff[2]) << 8 | static_cast<uint32_t>(buff[3]));         // NOLINT
  }

  // Return parsed integer
  return sign_num;
}

inline float ParseFloatLE(const char* buff)
{
  if (is_arch_LE())
  {
    float num{ *(reinterpret_cast<const float*>(buff)) };  // NOLINT
    return num;
  }

  uint32_t byte_float = 0;

  // Get correct unsigned value of float data
  byte_float = static_cast<uint32_t>(buff[3]) << 24 | static_cast<uint32_t>(buff[2]) << 16 |  // NOLINT
               static_cast<uint32_t>(buff[1]) << 8 | static_cast<uint32_t>(buff[0]);          // NOLINT

  // Return reinterpreted data
  return *(reinterpret_cast<float*>(&byte_float));  // NOLINT
}

inline float ParseFloatLE(const std::vector<char>& buff)
{
  if (is_arch_LE())
  {
    float num{ *(reinterpret_cast<const float*>(buff.data())) };  // NOLINT
    return num;
  }

  uint32_t byte_float = 0;

  // Get correct unsigned value of float data
  byte_float = static_cast<uint32_t>(buff[3]) << 24 | static_cast<uint32_t>(buff[2]) << 16 |  // NOLINT
               static_cast<uint32_t>(buff[1]) << 8 | static_cast<uint32_t>(buff[0]);          // NOLINT

  // Return reinterpreted data
  return *(reinterpret_cast<float*>(&byte_float));  // NOLINT
}

inline std::vector<std::string> SplitString(const std::string& s, char delimiter)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter))
  {
    tokens.push_back(token);
  }
  return tokens;
}

/**
 * Returns ingeger sampled from uniform distribution from the interval [from, to].
 */
template <typename T>
inline T rand_integral(T from, T to)
{
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<T> randFromDistribution(from, to);

  return randFromDistribution(rng);
}

inline float strToFloat(const std::string& str)
{
  float result;

  std::stringstream ss{ str };

  ss >> result;
  return result;
}

/**
 * Converts provided string into the `T` type.
 *
 * If not convertible we return defaultly constructed value of T.
 * Therefore we assume T to be default constructible.
 */
template <typename T>
inline T strTo(const std::string& str)
{
  T result;

  // Convert and check if successful
  if (auto [p, ec] = std::from_chars(str.data(), str.data() + str.size(), result); ec == std::errc())
  {
    return result;
  }
  // If failed
  else
  {
    std::string msg{ "Conversion of string '"s + str + "' failed with error code "s + std::to_string((int)ec) + "." };
    LOGE(msg);
    PROD_THROW("Data error!");
  }
}

[[nodiscard]] inline std::vector<std::string> tokenize_query_and(std::string_view query)
{
  char separator_char = '&';

  // Prepare sstream for parsing from it
  std::stringstream query_ss(query.data());

  std::vector<std::string> result_tokens;
  {
    std::string token_buffer;
    while (std::getline(query_ss, token_buffer, separator_char))
    {
      if (token_buffer.empty())
      {
        continue;
      }

      // Push new token into the result
      result_tokens.emplace_back(std::move(token_buffer));
    }
  }

  return result_tokens;
}
[[nodiscard]] inline std::string encode_and_query(const std::string& query)
{
  auto word_IDs(tokenize_query_and(query));

  std::string result_encoded_query{ "&"s };

  for (auto&& ID : word_IDs)
  {
    result_encoded_query += "-"s + ID + "+"s;
  }

  return result_encoded_query;
}

inline float dist_manhattan(const Vector<float>& left, const Vector<float>& right)
{
  float res{ 0.0F };

  for (size_t i{ 0_z }; i < left.size(); ++i)
  {
    float sub = std::abs(left[i] - right[i]);
    res += sub;
  }

  return res;
}

inline float dist_cos(const Vector<float>& left, const Vector<float>& right)
{
  float res{ 0.0F };

  float sum_l{ 0.0F };
  float sum_r{ 0.0F };

  for (size_t i{ 0_z }; i < left.size(); ++i)
  {
    sum_l += left[i] * left[i];
    sum_r += right[i] * right[i];

    res += left[i] * right[i];
  }

  res = res / (sqrtf(sum_l * sum_r));

  // Make is cosine distance
  return 1 - res;
}

inline float dist_cos_norm(const Vector<float>& left, const Vector<float>& right)
{
  float res{ 0.0F };

  for (size_t i{ 0_z }; i < left.size(); ++i)
  {
    res += left[i] * right[i];
  }

  // Make is cosine distance
  return 1 - res;
}

inline float dist_sq_eucl(const Vector<float>& left, const Vector<float>& right)
{
  float res{ 0.0F };

  for (size_t i{ 0_z }; i < left.size(); ++i)
  {
    float sub = left[i] - right[i];
    res += sub * sub;
  }

  return res;
}

inline float dist_eucl(const Vector<float>& left, const Vector<float>& right)
{
  return sqrtf(dist_sq_eucl(left, right));
}

inline std::function<float(const Vector<float>&, const Vector<float>&)> get_dist_fn(eDistFunction fn_type)
{
  switch (fn_type)
  {
    case eDistFunction::COSINE:
      return dist_cos;

    case eDistFunction::COSINE_NONORM:
      return dist_cos_norm;

    case eDistFunction::EUCLID:
      return dist_eucl;

    case eDistFunction::EUCLID_SQUARED:
      return dist_sq_eucl;

    case eDistFunction::MANHATTAN:
      return dist_manhattan;

    default:
      LOGW("Uknown distance function type: " + std::to_string(int(fn_type)) +
           "\n\tUsing default one: " + std::to_string(int(eDistFunction::MANHATTAN)));
      return dist_manhattan;
  }
}

template <typename CountType>
inline float tf_scheme_fn_natural(CountType freq_i, CountType max_freq)
{
  return float(freq_i) / max_freq;
}

template <typename CountType>
inline float tf_scheme_fn_logarithmic(CountType freq_i, CountType max_freq)
{
  return log(1.0F + (float(freq_i) / max_freq));
}

template <typename CountType>
inline float tf_scheme_fn_augmented(CountType freq_i, CountType max_freq)
{
  return 0.5F + 0.5F * (float(freq_i) / max_freq);  // NOLINT
}

inline std::function<float(float, float)> pick_tf_scheme_fn(eTermFrequency tf_scheme_ID)
{
  switch (tf_scheme_ID)
  {
    case eTermFrequency::NATURAL:
      return tf_scheme_fn_natural<float>;

    case eTermFrequency::LOGARIGHMIC:
      return tf_scheme_fn_logarithmic<float>;

    case eTermFrequency::AUGMENTED:
      return tf_scheme_fn_augmented<float>;

    default:
      LOGW("Unsupported scheme ID: " + std::to_string(int(tf_scheme_ID)) +
           "\n\t Using default: " + std::to_string(int(eTermFrequency::NATURAL)));
      return tf_scheme_fn_natural<float>;
  }
}

template <typename CountType>
inline float idf_scheme_fn_none([[maybe_unused]] CountType n_i, [[maybe_unused]] CountType N)
{
  return 1.0F;
}

template <typename CountType>
inline float idf_scheme_fn_IDF(CountType n_i, CountType N)
{
  return log(float(N) / n_i);
}

inline std::function<float(float, float)> pick_idf_scheme_fn(eInvDocumentFrequency idf_scheme_ID)
{
  switch (idf_scheme_ID)
  {
    case eInvDocumentFrequency::NONE:
      return idf_scheme_fn_none<float>;

    case eInvDocumentFrequency::IDF:
      return idf_scheme_fn_IDF<float>;

    default:
      LOGW("Unsupported scheme ID: " + std::to_string(int(idf_scheme_ID)) +
           "\n\t Using default: " + std::to_string(int(eInvDocumentFrequency::NONE)));
      return idf_scheme_fn_none<float>;
  }
}

template <typename T>
inline std::vector<T> vec_sub(const std::vector<T>& left, const std::vector<T>& right)
{
  if (left.size() != right.size())
  {
    throw std::runtime_error("Vectors have different sizes.");
  }

  std::vector<T> result;
  result.resize(left.size());

  uint32_t i = 0;
  for (auto& v : result)
  {
    v = left[i] - right[i];
    ++i;
  }

  return result;
}

template <typename T>
inline std::vector<T> vec_add(const std::vector<T>& left, const std::vector<T>& right)
{
  if (left.size() != right.size())
  {
    throw std::runtime_error("Vectors have different sizes.");
  }

  std::vector<T> result;
  result.resize(left.size());

  uint32_t i = 0;
  for (auto& v : result)
  {
    v = left[i] + right[i];
    ++i;
  }

  return result;
}

template <typename T, typename S>
inline std::vector<T> vec_mult(const std::vector<T>& left, S right)
{
  std::vector<T> result(left.size());

  std::transform(left.begin(), left.end(), result.begin(), [right](const T& l) { return l * right; });

  return result;
}

template <typename T>
inline std::vector<T> vec_mult(const std::vector<T>& left, const std::vector<T>& right)
{
  if (left.size() != right.size())
  {
    throw std::runtime_error("Vectors have different sizes.");
  }

  std::vector<T> result;

  std::transform(left.begin(), left.end(), right.begin(), std::back_inserter(result),
                 [](const T& l, const T& r) { return l * r; });

  return result;
}

template <typename T>
inline T dot_prod(const std::vector<T>& left, const std::vector<T>& right)
{
  if (left.size() != right.size())
  {
    throw std::runtime_error("Vectors have different sizes.");
  }

  std::vector<T> sum = vec_mult<T>(left, right);

  return std::accumulate(sum.begin(), sum.end(), 0.0f, std::plus<T>());
}

template <typename T>
inline std::vector<T> mat_vec_prod(const std::vector<std::vector<T>>& mat, const std::vector<T>& vec)
{
  if (mat.empty() || mat[0].size() != vec.size())
  {
    throw std::runtime_error("Vectors have different sizes or is mat empty.");
  }

  std::vector<T> result;

  for (auto&& mat_row_vec : mat)
  {
    result.emplace_back(dot_prod(mat_row_vec, vec));
  }

  return result;
}

template <typename T>
inline float length(const std::vector<T>& left)
{
  return sqrtf(dot_prod(left, left));
}

template <typename T>
inline std::vector<T> normalize(const std::vector<T>& left)
{
  float vec_size = length(left);

  if (vec_size > 0.0f)
    return vec_mult(left, (1.0f / vec_size));
  else
    throw std::runtime_error("Zero vec");
}

template <typename T>
inline Matrix<T> normalize_matrix_rows(const Matrix<T>& orig_mat)
{
  Matrix<T> res;

  std::transform(orig_mat.begin(), orig_mat.end(), std::back_inserter(res), normalize<T>);

  return res;
}

template <typename T>
inline Matrix<T> normalize_matrix_rows(Matrix<T>&& orig_mat)
{
  std::transform(orig_mat.begin(), orig_mat.end(), orig_mat.begin(), normalize<T>);

  return orig_mat;
}