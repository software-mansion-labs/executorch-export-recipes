// Qwen3-ASR .pte end-to-end runner: preprocess -> per-window audio_encoder ->
// token_embedding/text_decoder prefill -> greedy decode. Mirrors the RNE task.
//
// usage: qwen3_asr_bench model.pte PREFIX_IDS SUFFIX_IDS WARMUP clip.f32...
//   *_IDS: comma-separated prompt token ids; clip.f32: raw float32 16 kHz mono.
// Prints one line per clip: timings and the generated token ids
// (tools/device_wer.py turns a log into transcripts). Q3_ENC_ONLY=1 loads and
// times only preprocess + audio_encoder.
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::runtime::EValue;

namespace {

constexpr int kMel = 128;
constexpr int kWindowFrames = 800;
constexpr int kChunkFrames = 100;
constexpr int kMaxContext = 1024;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

std::vector<int64_t> parse_ids(const std::string& s) {
  std::vector<int64_t> ids;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) ids.push_back(std::stoll(tok));
  return ids;
}

std::vector<float> read_f32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<float> v(f.tellg() / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(float));
  return v;
}

int audio_token_count(int frames) {
  int rem = frames % kChunkFrames;
  int tail = 0;
  if (rem) {
    int a = (rem - 1) / 2 + 1;
    int b = (a - 1) / 2 + 1;
    tail = (b - 1) / 2 + 1;
  }
  return frames / kChunkFrames * 13 + tail;
}

#define CHECK_OK(expr, what)                                              \
  do {                                                                    \
    if (!(expr)) {                                                        \
      std::fprintf(stderr, "error: %s failed\n", what);                   \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

struct Runner {
  Module m;
  int dim = 0;
  explicit Runner(const std::string& path)
      : m(path, Module::LoadMode::MmapUseMlockIgnoreErrors) {}

  std::vector<float> embed(std::vector<int64_t> ids) {
    auto t = from_blob(ids.data(), {1, (int)ids.size()}, ScalarType::Long);
    auto r = m.execute("token_embedding", std::vector<EValue>{EValue(*t)});
    CHECK_OK(r.ok(), "token_embedding");
    auto out = r.get()[0].toTensor();
    dim = out.size(2);
    return std::vector<float>(out.const_data_ptr<float>(), out.const_data_ptr<float>() + out.numel());
  }

  int64_t decode_step(std::vector<float>& x, int64_t pos) {
    int64_t s = x.size() / dim;
    auto tx = from_blob(x.data(), {1, (int)s, dim}, ScalarType::Float);
    auto tp = from_blob(&pos, {1}, ScalarType::Long);
    auto r = m.execute("text_decoder", std::vector<EValue>{EValue(*tx), EValue(*tp)});
    CHECK_OK(r.ok(), "text_decoder");
    auto out = r.get()[0].toTensor();
    const float* p = out.const_data_ptr<float>();
    return std::max_element(p, p + out.numel()) - p;
  }

  void run(const std::string& name, std::vector<float> wav, const std::vector<int64_t>& prefix,
           const std::vector<int64_t>& suffix, bool print) {
    auto t0 = Clock::now();
    auto tw = from_blob(wav.data(), {(int)wav.size()}, ScalarType::Float);
    auto rp = m.execute("preprocess", std::vector<EValue>{EValue(*tw)});
    CHECK_OK(rp.ok(), "preprocess");
    auto mel_t = rp.get()[0].toTensor();
    int frames = mel_t.size(2);
    std::vector<float> mel(mel_t.const_data_ptr<float>(), mel_t.const_data_ptr<float>() + mel_t.numel());
    double pre_ms = ms_since(t0);

    auto t1 = Clock::now();
    std::vector<float> audio;
    int n_audio = 0;
    for (int w0 = 0; w0 < frames; w0 += kWindowFrames) {
      int f = std::min(kWindowFrames, frames - w0);
      int padded = (f + kChunkFrames - 1) / kChunkFrames * kChunkFrames;
      std::vector<float> buf((size_t)kMel * padded, 0.f);
      for (int b = 0; b < kMel; ++b)
        std::memcpy(&buf[(size_t)b * padded], &mel[(size_t)b * frames + w0], f * sizeof(float));
      int64_t n = audio_token_count(f);
      auto tb = from_blob(buf.data(), {1, kMel, padded}, ScalarType::Float);
      auto tn = from_blob(&n, {1}, ScalarType::Long);
      auto re = m.execute("audio_encoder", std::vector<EValue>{EValue(*tb), EValue(*tn)});
      CHECK_OK(re.ok(), "audio_encoder");
      auto out = re.get()[0].toTensor();
      int d = out.size(2);
      audio.insert(audio.end(), out.const_data_ptr<float>(), out.const_data_ptr<float>() + n * d);
      n_audio += n;
    }
    double enc_ms = ms_since(t1);
    if (std::getenv("Q3_ENC_ONLY") != nullptr) {
      if (print) std::printf("%s pre %.1f enc %.1f (%d audio tok)\n", name.c_str(), pre_ms, enc_ms, n_audio);
      std::fflush(stdout);
      return;
    }

    auto t2 = Clock::now();
    std::vector<float> x = embed(prefix);
    x.insert(x.end(), audio.begin(), audio.end());
    auto suf = embed(suffix);
    x.insert(x.end(), suf.begin(), suf.end());
    int64_t pos = x.size() / dim;
    int64_t tok = decode_step(x, 0);
    double prefill_ms = ms_since(t2);

    auto t3 = Clock::now();
    std::vector<int64_t> out;
    while (tok != 151643 && tok != 151645 && pos < kMaxContext) {
      out.push_back(tok);
      auto e = embed({tok});
      tok = decode_step(e, pos);
      ++pos;
    }
    double dec_ms = ms_since(t3);
    if (!print) return;
    std::printf("%s pre %.1f enc %.1f prefill %.1f (%lld tok) decode %.2f ms/tok (%zu) ids", name.c_str(),
                pre_ms, enc_ms, prefill_ms, (long long)(prefix.size() + n_audio + suffix.size()),
                out.empty() ? 0.0 : dec_ms / out.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i) std::printf("%s%lld", i ? "," : " ", (long long)out[i]);
    std::printf("\n");
    std::fflush(stdout);
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: %s model.pte PREFIX_IDS SUFFIX_IDS WARMUP clip.f32...\n", argv[0]);
    return 2;
  }
  auto prefix = parse_ids(argv[2]);
  auto suffix = parse_ids(argv[3]);
  int warmup = std::atoi(argv[4]);

  auto t0 = Clock::now();
  Runner r(argv[1]);
  std::vector<const char*> methods = {"preprocess", "audio_encoder"};
  if (std::getenv("Q3_ENC_ONLY") == nullptr) {
    methods.push_back("token_embedding");
    methods.push_back("text_decoder");
  }
  for (const char* meth : methods)
    CHECK_OK(r.m.load_method(meth) == executorch::runtime::Error::Ok, meth);
  std::printf("load %.1f ms\n", ms_since(t0));

  for (int i = 0; i < warmup; ++i) {
    auto w = Clock::now();
    r.run("warmup", read_f32(argv[5]), prefix, suffix, false);
    std::printf("warmup %d %.1f ms\n", i, ms_since(w));
  }
  for (int i = 5; i < argc; ++i) {
    std::string path = argv[i];
    r.run(path.substr(path.find_last_of('/') + 1), read_f32(path), prefix, suffix, true);
  }
  return 0;
}
