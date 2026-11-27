// Echo voice cloning on ExecuTorch: codec encoder -> encode_prefix -> sNcb2+2 sampler -> codec decoder.
// A port of run/voice_cloning.ipynb (and ~/echo-work/scripts/echo_ref.py).
#pragma once

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

namespace echo {

// export/xnnpack (harness) and export_encoder.py file names
inline constexpr const char* kDynamicDecoder = "codec_decoder_xnnpack_fp32.pte";
inline constexpr const char* kEncoder = "codec_encoder_xnnpack_fp32_multi.pte";

constexpr int kSampleRate = 24000;
constexpr int kHop = 960;
constexpr int kCodebooks = 8;
constexpr int kVocab = 1024;
constexpr int64_t kMask = 1024;

struct Stats {
  double encode_ref_ms = 0, prefix_ms = 0, sample_ms = 0, decode_ms = 0;
  int ref_frames = 0, prefix_len = 0, frames = 0, passes = 0;
  std::string backend;
  std::string to_string() const;
};

class Engine {
 public:
  // dir holds tokenizer.json, the backbone(s), kEncoder and the decoder(s).
  // A bucketed backbone (QNN, static Core ML) has fixed P <= PB, T <= TB; requests that do not fit it
  // run on `fallback` (a dynamic-shape backbone, e.g. Vulkan) when one is given.
  // decoder_frames > 0: the decoder is a static-shape export of that many frames (codes are padded with
  // the last frame, the audio is cut back). encoder "" skips the codec encoder (set_reference_codes only).
  std::string load(const std::string& dir, const std::string& backbone, const std::string& fallback = "",
                   const std::string& decoder = kDynamicDecoder,
                   const std::string& encoder = kEncoder, int decoder_frames = 0);

  // Reference given as codec codes [8, frames] (benchmarks, or a voice encoded earlier).
  void set_reference_codes(const std::vector<int64_t>& codes, int frames, int speech_frames);

  // wav: mono f32 at 24 kHz. Finds the speech, rounds it up to whole seconds (3..10) and encodes it.
  std::string set_reference(const std::vector<float>& wav, std::vector<float>* used = nullptr);

  // frames: target length from estimate_frames (computed by the caller, which has Unicode categories).
  std::string generate(const std::string& ref_text, const std::string& text, const std::string& lang,
                       int frames, uint64_t seed, std::vector<float>* audio,
                       const std::function<void(const std::string&)>& progress = nullptr);

  const Stats& stats() const { return stats_; }
  int ref_frames() const { return ref_frames_; }
  // Frames of actual speech in the reference (without the padding): use this for estimate_frames.
  int speech_frames() const { return speech_frames_; }

 private:
  struct Backbone {
    std::unique_ptr<executorch::extension::Module> m;
    std::string name;
    bool bucket = false;  // static padded bucket (QNN, static Core ML): key_bias/self_bias masks
    bool int32 = false;   // integer inputs are int32 (QNN) rather than int64
    int PB = 0, TB = 0;   // bucket sizes
    bool split = false;   // denoise -> (pred, conf) of codebooks 0-1; denoise_full -> raw logits
    std::vector<int> small;  // extra smaller prefix buckets, methods <name>_p<N> (same weights)
    std::string sfx;         // method suffix of the bucket the current request runs in
  };
  std::string open_backbone(const std::string& dir, const std::string& file, Backbone* bb);
  // Fills lp with [8, T, 1024] scores: log-probs, or with a split head, raw logits (every read of
  // those is an argmax a per-row constant does not move). With pred/conf and a split head, runs
  // the main-pass head instead and fills only those ([2, T]: argmax and its log-prob).
  std::string denoise(const std::vector<int64_t>& tokens, int T, int P, float ref_bias, std::vector<float>* lp,
                      std::vector<int64_t>* pred = nullptr, std::vector<float>* conf = nullptr);

  Backbone main_, fallback_;
  Backbone* cur_ = nullptr;
  std::unique_ptr<executorch::extension::Module> enc_, dec_;
  std::unique_ptr<executorch::extension::Module> dec_dyn_;  // with a static decoder: XNNPACK one for longer clips
  tokenizers::HFTokenizer tok_;
  std::vector<int64_t> ref_codes_;  // [8, ref_frames_]
  int ref_frames_ = 0, nr_ = 0, speech_frames_ = 0, kv_len_ = 0, decoder_frames_ = 0;
  bool dec_int32_ = false;
  std::vector<std::vector<uint8_t>> kv_;  // 48 x [6, P, 128], raw bytes of kv_type_
  executorch::aten::ScalarType kv_type_ = executorch::aten::ScalarType::Float;  // fp16 on builds exported with --kv16
  std::vector<std::vector<int64_t>> kv_sizes_;
  std::string enc_path_;
  Stats stats_;
};

std::pair<size_t, size_t> silence_bounds(const std::vector<float>& audio, int sample_rate, float keep);
std::vector<float> trim_silence(const std::vector<float>& audio, int sample_rate, float keep = 0.1f);

}  // namespace echo
