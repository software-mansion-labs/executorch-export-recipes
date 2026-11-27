#include "echo_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include <executorch/extension/tensor/tensor.h>

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace echo {
namespace {

constexpr float kTShift = 0.1f, kCodebookPenalty = 5.0f, kPositionTemperature = 5.0f, kRemaskRatio = 0.5f;
constexpr float kPolishBias = 1.0f, kPolishGuidance = 0.5f;
constexpr int kSteps = 8;

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string err(const char* what, Error e) {
  char b[160];
  std::snprintf(b, sizeof b, "%s failed (error 0x%x)", what, (int)e);
  return b;
}

}  // namespace

std::string Stats::to_string() const {
  char b[600];
  std::snprintf(b, sizeof b,
                "backbone: %s\nreference: %d frames (%.1f s), encoded in %.0f ms\nprefix: %d tokens, %.0f ms\n"
                "sampling: %d frames (%.1f s of speech), %d passes, %.0f ms\ncodec decode: %.0f ms",
                backend.c_str(), ref_frames, ref_frames / 25.0, encode_ref_ms, prefix_len, prefix_ms, frames, frames / 25.0, passes,
                sample_ms, decode_ms);
  return b;
}

std::string Engine::open_backbone(const std::string& dir, const std::string& file, Backbone* bb) {
  bb->m = std::make_unique<Module>(dir + "/" + file, Module::LoadMode::Mmap);
  bb->name = file;
  auto names = bb->m->method_names();
  bb->split = names.ok() && names->count("denoise_full");
  for (const char* name : {"encode_prefix", "denoise", "denoise_full"}) {
    if (!strcmp(name, "denoise_full") && !bb->split) continue;
    Error e = bb->m->load_method(name);
    if (e != Error::Ok) return err(("load " + file + ":" + name).c_str(), e);
  }
  auto pre = bb->m->method_meta("encode_prefix");
  auto den = bb->m->method_meta("denoise");
  if (!pre.ok() || !den.ok()) return "cannot read method meta of " + file;
  auto in0 = pre->input_tensor_meta(0);
  bb->int32 = in0.ok() && in0->scalar_type() == ScalarType::Int;
  bb->bucket = pre->num_inputs() == 5;  // encode_prefix(text, audio, is_audio, pos, key_bias)
  if (bb->bucket && names.ok()) {
    for (const auto& n : *names) {  // encode_prefix_p<N>: a smaller prefix bucket
      if (n.rfind("encode_prefix_p", 0) != 0) continue;
      const std::string sfx = n.substr(strlen("encode_prefix"));
      for (const char* base : {"encode_prefix", "denoise", "denoise_full"}) {
        if (!strcmp(base, "denoise_full") && !bb->split) continue;
        Error e = bb->m->load_method(base + sfx);
        if (e != Error::Ok) return err(("load " + file + ":" + base + sfx).c_str(), e);
      }
      bb->small.push_back(std::stoi(sfx.substr(2)));
    }
    std::sort(bb->small.begin(), bb->small.end());
  }
  if (bb->bucket) {
    bb->PB = (int)in0->sizes()[0];
    bb->TB = (int)den->input_tensor_meta(0)->sizes()[1];
  }
  return "";
}

std::string Engine::load(const std::string& dir, const std::string& backbone, const std::string& fallback,
                         const std::string& decoder, const std::string& encoder, int decoder_frames) {
  decoder_frames_ = decoder_frames;
  if (tok_.load(dir + "/tokenizer.json") != tokenizers::Error::Ok) return "cannot load tokenizer.json";
  auto open = [&](std::unique_ptr<Module>& m, const std::string& file, const std::vector<std::string>& methods) {
    m = std::make_unique<Module>(dir + "/" + file, Module::LoadMode::Mmap);
    for (auto& name : methods) {
      Error e = m->load_method(name);
      if (e != Error::Ok) return err(("load " + file + ":" + name).c_str(), e);
    }
    return std::string();
  };
  std::string e = open_backbone(dir, backbone, &main_);
  if (e.empty() && !fallback.empty() && fallback != backbone) e = open_backbone(dir, fallback, &fallback_);
  if (e.empty()) e = open(dec_, decoder, {"decode"});
  if (e.empty()) {  // QNN decoders take int32 codes (the HTP has no int64 tensors)
    auto in = dec_->method_meta("decode")->input_tensor_meta(0);
    dec_int32_ = in.ok() && in->scalar_type() == ScalarType::Int;
  }
  if (e.empty() && decoder_frames > 0 && decoder != kDynamicDecoder) e = open(dec_dyn_, kDynamicDecoder, {"decode"});
  // The codec encoder (200M fp32 params, ~1.5 GB with XNNPACK's packed copy) runs once per recording:
  // set_reference opens it and frees it again.
  enc_path_ = encoder.empty() ? "" : dir + "/" + encoder;
  return e;
}

std::pair<size_t, size_t> silence_bounds(const std::vector<float>& audio, int sample_rate, float keep) {
  // Leading/trailing 10 ms windows under -50 dBFS; isolated edge blips of <= 120 ms dropped (notebook rule).
  const int window = sample_rate / 100;
  const int count = (int)audio.size() / window;
  std::vector<int> loud;
  for (int i = 0; i < count; i++) {
    double s = 0;
    for (int j = 0; j < window; j++) s += (double)audio[i * window + j] * audio[i * window + j];
    if (20 * std::log10(std::sqrt(s / window) + 1e-10) >= -50) loud.push_back(i);
  }
  if (loud.empty()) return {0, audio.size()};
  std::vector<std::pair<int, int>> sounds;  // [first, last] window index
  int start = loud[0];
  for (size_t i = 1; i <= loud.size(); i++) {
    if (i == loud.size() || loud[i] - loud[i - 1] > 20) {
      sounds.push_back({start, loud[i - 1]});
      if (i < loud.size()) start = loud[i];
    }
  }
  auto len = [](const std::pair<int, int>& s) { return s.second - s.first + 1; };
  if (sounds.size() > 1 && len(sounds.front()) <= 12) sounds.erase(sounds.begin());
  if (sounds.size() > 1 && len(sounds.back()) <= 12) sounds.pop_back();
  const int margin = (int)std::lround(keep * 100);
  const size_t a = (size_t)std::max(sounds.front().first - margin, 0) * window;
  const size_t b = std::min(audio.size(), (size_t)(sounds.back().second + 1 + margin) * window);
  return {a, b};
}

std::vector<float> trim_silence(const std::vector<float>& audio, int sample_rate, float keep) {
  auto [a, b] = silence_bounds(audio, sample_rate, keep);
  return std::vector<float>(audio.begin() + a, audio.begin() + b);
}

void Engine::set_reference_codes(const std::vector<int64_t>& codes, int frames, int speech_frames) {
  ref_codes_ = codes;
  ref_frames_ = frames;
  speech_frames_ = speech_frames;
  stats_.ref_frames = frames;
  stats_.encode_ref_ms = 0;
}

std::string Engine::set_reference(const std::vector<float>& wav, std::vector<float>* used) {
  const double t0 = now_ms();
  // The encoder has one method per whole second (3..10 s). Round the speech up to the next one,
  // filling with the recording's own tail and then silence, so no words are cut off.
  auto [a, b] = silence_bounds(wav, kSampleRate, 0.1f);
  const double speech = (double)(b - a) / kSampleRate;
  if (speech < 1.0) return "no speech found in the recording";
  if (speech > 10.0) return "the speech is longer than 10 s; record 3 to 10 s";
  const int seconds = std::max(3, (int)std::ceil(speech));
  speech_frames_ = (int)std::lround(speech * kSampleRate / kHop);
  std::vector<float> x((size_t)seconds * kSampleRate, 0.f);
  const size_t take = std::min(x.size(), wav.size() - a);
  std::copy(wav.begin() + a, wav.begin() + a + take, x.begin());
  if (enc_path_.empty()) return "no codec encoder configured";
  enc_ = std::make_unique<Module>(enc_path_, Module::LoadMode::Mmap);
  struct Release { std::unique_ptr<Module>& m; ~Release() { m.reset(); } } release{enc_};
  const std::string method = "encode_" + std::to_string(seconds) + "s";
  auto in = from_blob(x.data(), {(int)x.size()}, ScalarType::Float);
  auto r = enc_->execute(method, std::vector<EValue>{*in});
  if (!r.ok()) return err(("codec " + method).c_str(), r.error());
  auto codes = r.get()[0].toTensor();
  ref_frames_ = (int)codes.size(1);
  ref_codes_.assign(codes.const_data_ptr<int64_t>(), codes.const_data_ptr<int64_t>() + codes.numel());
  stats_.encode_ref_ms = now_ms() - t0;
  stats_.ref_frames = ref_frames_;
  if (used) *used = std::move(x);
  return "";
}

std::string Engine::denoise(const std::vector<int64_t>& tokens, int T, int P, float ref_bias, std::vector<float>* lp,
                            std::vector<int64_t>* pred, std::vector<float>* conf) {
  constexpr float kNeg = -1e4f;  // additive mask on padded keys (QNN buckets)
  const bool bucketed = cur_->bucket, i32 = cur_->int32;
  const int TB = bucketed ? cur_->TB : T, PK = kv_len_;
  std::vector<TensorPtr> keep;
  std::vector<int64_t> codes64, pos64;
  std::vector<int32_t> codes32, pos32;
  std::vector<float> bias(PK, 0.f), self_bias;
  for (int i = P - nr_; i < P; i++) bias[i] = ref_bias;
  if (bucketed) {
    for (int i = P; i < PK; i++) bias[i] = kNeg;
    codes64.assign((size_t)kCodebooks * TB, kMask);
    pos64.assign(TB, 0);
    self_bias.assign(TB, kNeg);
    for (int c = 0; c < kCodebooks; c++)
      for (int t = 0; t < T; t++) codes64[(size_t)c * TB + t] = tokens[(size_t)c * T + t];
    for (int t = 0; t < T; t++) pos64[t] = P + t, self_bias[t] = 0.f;
    if (i32) {
      codes32.assign(codes64.begin(), codes64.end());
      pos32.assign(pos64.begin(), pos64.end());
      keep.push_back(from_blob(codes32.data(), {kCodebooks, TB}, ScalarType::Int));
      keep.push_back(from_blob(pos32.data(), {TB}, ScalarType::Int));
    } else {
      keep.push_back(from_blob(codes64.data(), {kCodebooks, TB}, ScalarType::Long));
      keep.push_back(from_blob(pos64.data(), {TB}, ScalarType::Long));
    }
    keep.push_back(from_blob(bias.data(), {PK}, ScalarType::Float));
    keep.push_back(from_blob(self_bias.data(), {TB}, ScalarType::Float));
  } else {
    pos64.resize(T);
    std::iota(pos64.begin(), pos64.end(), (int64_t)P);
    keep.push_back(from_blob(const_cast<int64_t*>(tokens.data()), {kCodebooks, T}, ScalarType::Long));
    keep.push_back(from_blob(pos64.data(), {T}, ScalarType::Long));
    keep.push_back(from_blob(bias.data(), {PK}, ScalarType::Float));
  }
  for (size_t i = 0; i < kv_.size(); i++) {
    std::vector<int> sz(kv_sizes_[i].begin(), kv_sizes_[i].end());
    keep.push_back(from_blob(kv_[i].data(), sz, kv_type_));
  }
  std::vector<EValue> in;
  for (auto& t : keep) in.emplace_back(*t);
  const bool main_head = cur_->split && pred;
  const std::string method = std::string(cur_->split && !pred ? "denoise_full" : "denoise") + cur_->sfx;
  auto r = cur_->m->execute(method, in);
  if (!r.ok()) return err(method.c_str(), r.error());
  stats_.passes++;
  if (main_head) {  // ([2, TB] argmax, [2, TB] log-prob) -> [2, T] each
    const auto& p = r.get()[0].toTensor();
    const float* cf = r.get()[1].toTensor().const_data_ptr<float>();
    pred->resize((size_t)2 * T);
    conf->resize((size_t)2 * T);
    for (int c = 0; c < 2; c++)
      for (int t = 0; t < T; t++) {
        const size_t i = (size_t)c * TB + t, o = (size_t)c * T + t;
        (*pred)[o] = p.scalar_type() == ScalarType::Int ? p.const_data_ptr<int32_t>()[i] : p.const_data_ptr<int64_t>()[i];
        (*conf)[o] = cf[i];
      }
    return "";
  }
  const float* out = r.get()[0].toTensor().const_data_ptr<float>();
  if (cur_->split) {  // [TB, 8 * 1025] raw logits, token-major -> [8, T, 1024]
    lp->resize((size_t)kCodebooks * T * kVocab);
    for (int t = 0; t < T; t++)
      for (int c = 0; c < kCodebooks; c++)
        std::copy(out + ((size_t)t * kCodebooks + c) * (kVocab + 1), out + ((size_t)t * kCodebooks + c) * (kVocab + 1) + kVocab,
                  lp->data() + ((size_t)c * T + t) * kVocab);
    return "";
  }
  lp->resize((size_t)kCodebooks * T * kVocab);
  for (int c = 0; c < kCodebooks; c++)  // [8, TB, 1024] -> [8, T, 1024]
    std::copy(out + (size_t)c * TB * kVocab, out + ((size_t)c * TB + T) * kVocab, lp->data() + (size_t)c * T * kVocab);
  return "";
}

std::string Engine::generate(const std::string& ref_text, const std::string& text, const std::string& lang,
                             int frames, uint64_t seed, std::vector<float>* audio,
                             const std::function<void(const std::string&)>& progress) {
  if (ref_frames_ == 0) return "no reference recorded";
  auto say = [&](const std::string& s) { if (progress) progress(s); };
  stats_.passes = 0;

  // ---- prefix
  double t0 = now_ms();
  const std::string prompt = "<|denoise|><|lang_start|>" + lang + "<|lang_end|><|instruct_start|>None<|instruct_end|>"
                             "<|text_start|>" + ref_text + " " + text + "<|text_end|>";
  auto ids = tok_.encode(prompt, 0, 0);
  if (!ids.ok()) return "tokenizer failed";
  const int nt = (int)ids.get().size();
  nr_ = ref_frames_;
  const int P = nt + nr_;
  const int T = frames;
  cur_ = &main_;
  stats_.backend = main_.name;
  if (main_.bucket && (P > main_.PB || T > main_.TB)) {
    if (!fallback_.m)
      return "too long for the NPU bucket (prompt " + std::to_string(P) + "/" + std::to_string(main_.PB) +
             ", frames " + std::to_string(T) + "/" + std::to_string(main_.TB) + "): shorten the texts";
    cur_ = &fallback_;
    stats_.backend = fallback_.name + " (text too long for the NPU bucket)";
  }
  if (!cur_->bucket && P > 768) return "prompt too long (" + std::to_string(P) + " > 768 tokens): shorten the texts";
  int PK = cur_->bucket ? cur_->PB : P;  // padded prefix length
  cur_->sfx.clear();
  for (int pb : cur_->small)  // the smallest bucket that fits: attention and the prefix pass scale with it
    if (cur_->bucket && P <= pb) {
      PK = pb;
      cur_->sfx = "_p" + std::to_string(pb);
      break;
    }
  std::vector<int64_t> text_in(PK, 0), audio_in((size_t)kCodebooks * PK, 0);
  std::vector<float> is_audio(PK, 0.f), key_bias(PK, 0.f);
  for (int i = 0; i < nt; i++) text_in[i] = (int64_t)ids.get()[i];
  for (int c = 0; c < kCodebooks; c++)
    for (int j = 0; j < nr_; j++) audio_in[(size_t)c * PK + nt + j] = ref_codes_[(size_t)c * nr_ + j];
  for (int j = nt; j < P; j++) is_audio[j] = 1.f;
  for (int j = P; j < PK; j++) key_bias[j] = -1e4f;
  std::vector<TensorPtr> keep;
  std::vector<int32_t> text32, audio32, pos32;
  std::vector<int64_t> pos64;
  if (cur_->bucket && cur_->int32) {
    text32.assign(text_in.begin(), text_in.end());
    audio32.assign(audio_in.begin(), audio_in.end());
    pos32.resize(PK);
    std::iota(pos32.begin(), pos32.end(), 0);
    keep = {from_blob(text32.data(), {PK}, ScalarType::Int), from_blob(audio32.data(), {kCodebooks, PK}, ScalarType::Int),
            from_blob(is_audio.data(), {PK}, ScalarType::Float), from_blob(pos32.data(), {PK}, ScalarType::Int),
            from_blob(key_bias.data(), {PK}, ScalarType::Float)};
  } else if (cur_->bucket) {
    pos64.resize(PK);
    std::iota(pos64.begin(), pos64.end(), 0);
    keep = {from_blob(text_in.data(), {PK}, ScalarType::Long),
            from_blob(audio_in.data(), {kCodebooks, PK}, ScalarType::Long),
            from_blob(is_audio.data(), {PK}, ScalarType::Float), from_blob(pos64.data(), {PK}, ScalarType::Long),
            from_blob(key_bias.data(), {PK}, ScalarType::Float)};
  } else {
    pos64.resize(PK);
    std::iota(pos64.begin(), pos64.end(), 0);
    keep = {from_blob(text_in.data(), {PK}, ScalarType::Long),
            from_blob(audio_in.data(), {kCodebooks, PK}, ScalarType::Long),
            from_blob(is_audio.data(), {PK}, ScalarType::Float), from_blob(pos64.data(), {PK}, ScalarType::Long)};
  }
  std::vector<EValue> pin;
  for (auto& t : keep) pin.emplace_back(*t);
  auto r = cur_->m->execute("encode_prefix" + cur_->sfx, pin);
  if (!r.ok()) return err(("encode_prefix" + cur_->sfx).c_str(), r.error());
  kv_.clear();
  kv_sizes_.clear();
  kv_len_ = PK;
  for (size_t i = 0; i < 48; i++) {  // a16w8 QNN builds return the last hidden state as a 49th output
    auto t = r.get()[i].toTensor();
    const auto* p = static_cast<const uint8_t*>(t.const_data_ptr());
    kv_.emplace_back(p, p + t.nbytes());
    kv_type_ = t.scalar_type();
    kv_sizes_.emplace_back(t.sizes().begin(), t.sizes().end());
  }
  stats_.prefix_ms = now_ms() - t0;
  stats_.prefix_len = P;

  // ---- sampler: sNcb2 + 2 (8 main passes on codebooks 0-1, then two polish passes)
  t0 = now_ms();
  stats_.frames = T;
  std::vector<int64_t> tokens((size_t)kCodebooks * T, kMask);
  std::vector<float> lp, conf((size_t)2 * T);
  std::vector<int64_t> pred((size_t)2 * T);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  std::vector<double> ts(kSteps + 1);
  for (int i = 0; i <= kSteps; i++) {
    double t = (double)i / kSteps;
    ts[i] = kTShift * t / (1 - (1 - kTShift) * t);
  }
  for (int step = 0; step < kSteps; step++) {
    say("pass " + std::to_string(step + 1) + " / " + std::to_string(kSteps + 2));
    std::string e = cur_->split ? denoise(tokens, T, P, 0.f, &lp, &pred, &conf) : denoise(tokens, T, P, 0.f, &lp);
    if (!e.empty()) return e;
    std::vector<int> masked, revealed;
    for (int c = 0; c < 2; c++)
      for (int t = 0; t < T; t++) {
        const size_t i = (size_t)c * T + t;
        if (!cur_->split) {
          const float* row = lp.data() + ((size_t)c * T + t) * kVocab;
          int best = (int)(std::max_element(row, row + kVocab) - row);
          conf[i] = row[best];
          pred[i] = best;
        }
        (tokens[i] == kMask ? masked : revealed).push_back((int)i);
      }
    const bool last = step == kSteps - 1;
    const int k = last ? (int)masked.size()
                       : std::min((int)masked.size(), (int)std::ceil(2.0 * T * (ts[step + 1] - ts[step])));
    std::vector<std::pair<float, int>> score;
    score.reserve(masked.size());
    for (int i : masked) {
      const int c = i / T;
      float g = -std::log(-std::log(uni(rng) + 1e-10f) + 1e-10f);
      score.push_back({(conf[i] - kCodebookPenalty * c) / kPositionTemperature + g, i});
    }
    std::partial_sort(score.begin(), score.begin() + k, score.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    for (int j = 0; j < k; j++) tokens[score[j].second] = pred[score[j].second];
    if (!last) {
      // Python round() is round-half-to-even, as is nearbyint in the default rounding mode.
      const int count = std::min((int)std::nearbyint(kRemaskRatio * k), (int)revealed.size());
      std::vector<std::pair<float, int>> low;
      for (int i : revealed) low.push_back({conf[i], i});
      std::partial_sort(low.begin(), low.begin() + count, low.end(),
                        [](auto& a, auto& b) { return a.first < b.first; });
      for (int j = 0; j < count; j++) tokens[low[j].second] = kMask;
    }
  }
  say("polish pass 1");
  std::vector<float> first, second;
  std::string e = denoise(tokens, T, P, kPolishBias, &first);
  if (!e.empty()) return e;
  auto argmax = [&](const std::vector<float>& a, int c, int t) {
    const float* row = a.data() + ((size_t)c * T + t) * kVocab;
    return (int64_t)(std::max_element(row, row + kVocab) - row);
  };
  for (int c = 2; c < 4; c++)
    for (int t = 0; t < T; t++) tokens[(size_t)c * T + t] = argmax(first, c, t);
  say("polish pass 2");
  e = denoise(tokens, T, P, kPolishBias, &second);
  if (!e.empty()) return e;
  for (int c = 4; c < kCodebooks; c++)
    for (int t = 0; t < T; t++) {
      const float* s = second.data() + ((size_t)c * T + t) * kVocab;
      const float* f = first.data() + ((size_t)c * T + t) * kVocab;
      int best = 0;
      float bv = -INFINITY;
      for (int v = 0; v < kVocab; v++) {
        float g = s[v] + kPolishGuidance * (s[v] - f[v]);
        if (g > bv) bv = g, best = v;
      }
      tokens[(size_t)c * T + t] = best;
    }
  stats_.sample_ms = now_ms() - t0;

  // ---- codec decode, drop the last 0.2 s, trim silence
  say("decoding audio");
  t0 = now_ms();
  int TD = T;
  const bool dyn = decoder_frames_ > 0 && T > decoder_frames_ && dec_dyn_;  // longer than the static decoder
  if (decoder_frames_ > 0 && !dyn) {  // static decoder: pad with the last frame, cut the audio back below
    if (T > decoder_frames_) return "too many frames for the static decoder";
    std::vector<int64_t> padded((size_t)kCodebooks * decoder_frames_);
    for (int c = 0; c < kCodebooks; c++)
      for (int t = 0; t < decoder_frames_; t++) padded[(size_t)c * decoder_frames_ + t] = tokens[(size_t)c * T + std::min(t, T - 1)];
    tokens.swap(padded);
    TD = decoder_frames_;
  }
  const bool i32 = dec_int32_ && !dyn;
  std::vector<int32_t> tokens32;
  if (i32) tokens32.assign(tokens.begin(), tokens.end());
  auto t_codes = i32 ? from_blob(tokens32.data(), {kCodebooks, TD}, ScalarType::Int)
                     : from_blob(tokens.data(), {kCodebooks, TD}, ScalarType::Long);
  auto d = (dyn ? dec_dyn_ : dec_)->execute("decode", std::vector<EValue>{*t_codes});
  if (!d.ok()) return err("codec decode", d.error());
  auto a = d.get()[0].toTensor();
  std::vector<float> wav(a.const_data_ptr<float>(), a.const_data_ptr<float>() + std::min<size_t>(a.numel(), (size_t)T * kHop));
  wav.resize(wav.size() - std::min(wav.size(), (size_t)(0.2 * kSampleRate)));
  *audio = trim_silence(wav, kSampleRate);
  stats_.decode_ms = now_ms() - t0;
  return "";
}

}  // namespace echo
