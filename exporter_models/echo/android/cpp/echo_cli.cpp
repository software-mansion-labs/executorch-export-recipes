// usage: echo_cli <model_dir> <backbone.pte> <ref_24k_f32.raw> <ref_text.txt> <text.txt> <lang> <frames> <seed> <out_f32.raw> [fallback.pte]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

#include "echo_engine.h"

static std::string slurp(const char* p) {
  std::ifstream f(p);
  std::stringstream s;
  s << f.rdbuf();
  std::string r = s.str();
  while (!r.empty() && (r.back() == '\n' || r.back() == '\r')) r.pop_back();
  return r;
}

int main(int argc, char** argv) {
  if (argc < 10) { std::fprintf(stderr, "usage\n"); return 2; }
  echo::Engine eng;
  // ECHO_DECODER / ECHO_DECODER_FRAMES pick another codec decoder, e.g. a static QNN one.
  const char* dec = std::getenv("ECHO_DECODER");
  const char* dec_frames = std::getenv("ECHO_DECODER_FRAMES");
  std::string e = dec ? eng.load(argv[1], argv[2], argc > 10 ? argv[10] : "", dec, echo::kEncoder,
                                 dec_frames ? std::atoi(dec_frames) : 0)
                      : eng.load(argv[1], argv[2], argc > 10 ? argv[10] : "");
  if (!e.empty()) { std::printf("LOAD_FAIL %s\n", e.c_str()); return 3; }
  std::ifstream rf(argv[3], std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(rf)), std::istreambuf_iterator<char>());
  std::vector<float> wav(raw.size() / 4);
  std::memcpy(wav.data(), raw.data(), wav.size() * 4);
  e = eng.set_reference(wav);
  if (!e.empty()) { std::printf("REF_FAIL %s\n", e.c_str()); return 4; }
  std::vector<float> audio;
  std::printf("SPEECH_FRAMES %d REF_FRAMES %d\n", eng.speech_frames(), eng.ref_frames());
  // frames <= 0: frames = -frames * speech_frames / 1000 (host passes weight ratio x1000)
  int frames = std::atoi(argv[7]);
  if (frames <= 0) frames = std::max(1, (int)(-frames / 1000.0 * eng.speech_frames()));
  e = eng.generate(slurp(argv[4]), slurp(argv[5]), argv[6], frames, std::strtoull(argv[8], nullptr, 10),
                   &audio, [](const std::string& s) { std::printf("  %s\n", s.c_str()); std::fflush(stdout); });
  if (!e.empty()) { std::printf("GEN_FAIL %s\n", e.c_str()); return 5; }
  std::ofstream o(argv[9], std::ios::binary);
  o.write((const char*)audio.data(), audio.size() * 4);
  std::printf("%s\nAUDIO %zu samples\nOK\n", eng.stats().to_string().c_str(), audio.size());
  return 0;
}
