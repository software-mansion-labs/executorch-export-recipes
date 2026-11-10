#include <algorithm>
#include <cmath>
#include <cstdio>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using Values = std::vector<EValue>;
template <class T> std::vector<T> read(const std::string &p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("open " + p);
  size_t n = f.tellg();
  std::vector<T> v(n / sizeof(T));
  f.seekg(0);
  f.read((char *)v.data(), n);
  return v;
}
Values run(Module &m, const char *name, const Values &input) {
  auto r = m.execute(name, input);
  if (!r.ok())
    throw std::runtime_error(std::string(name) + " error " +
                             std::to_string((int)r.error()));
  printf("EXEC %s\n", name);
  fflush(stdout);
  return std::move(*r);
}
void save(const std::string &p, const EValue &v) {
  const auto &t = v.toTensor();
  const float *data = t.const_data_ptr<float>();
  for (size_t i = 0; i < t.numel(); i++)
    if (!std::isfinite(data[i]))
      throw std::runtime_error("Non-finite " + p);
  std::ofstream f(p, std::ios::binary);
  f.write((char *)data, t.nbytes());
  printf("SAVE %s numel=%zu\n", p.c_str(), t.numel());
  fflush(stdout);
}
int main(int argc, char **argv) {
  try {
    if (argc < 3)
      return 1;
    executorch::runtime::runtime_init();
    Module m(argv[1], Module::LoadMode::Mmap);
    std::string dir = argv[2];
    if (argc > 3 && std::string(argv[3]) == "vision") {
      auto pixels = read<float>(dir + "/image.bin");
      auto image = executorch::extension::make_tensor_ptr<float>(
          {1, 3, 512, 512}, pixels);
      auto out = run(m, "vision_encoder", {*image});
      save(dir + "/actual-vision.bin", out[0]);
      puts("VISION PASS");
      return 0;
    }
    if (argc > 3 && std::string(argv[3]) == "components") {
      auto hv = read<float>(dir + "/depth-hidden.bin");
      auto cv = read<int64_t>(dir + "/depth-codes.bin");
      auto mv = read<uint8_t>(dir + "/depth-mask.bin");
      auto hidden =
          executorch::extension::make_tensor_ptr<float>({1, 2048}, hv);
      auto codes = executorch::extension::make_tensor_ptr<int64_t>({1, 12}, cv);
      auto mask = executorch::extension::make_tensor_ptr(
          {1, 12}, mv.data(), executorch::aten::ScalarType::Bool);
      auto depth = run(m, "depth_decoder", {*hidden, *codes, *mask});
      save(dir + "/actual-depth.bin", depth[0]);
      auto audioemb = run(m, "audio_embedding", {*codes});
      save(dir + "/actual-audio-embedding.bin", audioemb[0]);
      for (int n : {1, 4, 32}) {
        auto c = read<int64_t>(dir + "/codec-" + std::to_string(n) + ".bin");
        auto t = executorch::extension::make_tensor_ptr<int64_t>({1, 12, n}, c);
        auto out = run(m, "audio_decoder", {*t});
        save(dir + "/actual-codec-" + std::to_string(n) + ".bin", out[0]);
      }
      for (int n : {400, 16400, 80400}) {
        auto wave = read<float>(dir + "/wave-" + std::to_string(n) + ".bin");
        auto t = executorch::extension::make_tensor_ptr<float>({1, n}, wave);
        auto out = run(m, "audio_encoder", {*t});
        save(dir + "/actual-audio-" + std::to_string(n) + ".bin", out[0]);
      }
      puts("COMPONENTS PASS");
      return 0;
    }
    auto ids = read<int64_t>(dir + "/prompt.bin");
    auto token = executorch::extension::make_tensor_ptr<int64_t>(
        {1, (int32_t)ids.size()}, ids);
    auto emb = run(m, "token_embedding", {*token});
    save(dir + "/embedding.bin", emb[0]);
    std::vector<int64_t> positions(ids.size());
    std::iota(positions.begin(), positions.end(), 0);
    auto pos = executorch::extension::make_tensor_ptr<int64_t>(
        {(int32_t)ids.size()}, positions);
    auto out = run(m, "text_decoder", {emb[0], *pos});
    save(dir + "/prefill-logits.bin", out[0]);
    save(dir + "/prefill-hidden.bin", out[1]);
    int64_t position = ids.size();
    for (int step = 0; step < 32; step++) {
      auto &t = out[0].toTensor();
      const float *p = t.const_data_ptr<float>();
      int64_t next = 0;
      float best = -INFINITY;
      for (int64_t i = 0; i < t.numel(); i++) {
        if (i == 125029 || i == 125030 || i == 125031 || i == 125032 ||
            i == 125012 || i == 125015)
          continue;
        if (p[i] > best) {
          best = p[i];
          next = i;
        }
      }
      printf("TOKEN %lld\n", (long long)next);
      fflush(stdout);
      if (next == 124900 || next == 125014)
        break;
      token = executorch::extension::make_tensor_ptr<int64_t>({1, 1}, {next});
      emb = run(m, "token_embedding", {*token});
      pos = executorch::extension::make_tensor_ptr<int64_t>({1}, {position++});
      out = run(m, "text_decoder", {emb[0], *pos});
    }
    puts("PASS");
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "FAIL %s\n", e.what());
    return 2;
  }
}
