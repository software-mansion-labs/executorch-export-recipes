#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <fstream>
#include <iterator>
#include <sys/resource.h>
#include <vector>

using namespace executorch::extension;
using namespace executorch::runtime;

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "model.pte prompt.i64 output_prefix steps [chunk_size]\n");
    return 2;
  }
  runtime_init();
  std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
  const auto bytes = input.tellg();
  if (bytes <= 0 || bytes % sizeof(int64_t))
    return 3;
  input.seekg(0);
  std::vector<int64_t> prompt(bytes / sizeof(int64_t));
  input.read(reinterpret_cast<char *>(prompt.data()), bytes);
  if (!input)
    return 3;
  const int steps = std::atoi(argv[4]);
  const size_t chunk_size = argc > 5 ? std::atoi(argv[5]) : 128;
  if (chunk_size < 1 || chunk_size > 128)
    return 3;
  Module model(argv[1], Module::LoadMode::Mmap);
  auto loaded = model.load_method("forward");
  if (loaded != Error::Ok) {
    std::fprintf(stderr, "load_method failed: 0x%x\n", unsigned(loaded));
    return 4;
  }
  std::vector<float> logits;
  auto forward = [&](const int64_t *ids, int count, int64_t pos) {
    auto tokens = make_tensor_ptr({1, count}, const_cast<int64_t *>(ids),
                                  executorch::aten::ScalarType::Long);
    auto position =
        make_tensor_ptr({1}, &pos, executorch::aten::ScalarType::Long);
    auto output =
        model.execute("forward", std::vector<EValue>{*tokens, *position});
    if (!output.ok()) {
      std::fprintf(stderr, "forward failed at pos %lld, length %d: 0x%x\n",
                   (long long)pos, count, unsigned(output.error()));
      return false;
    }
    if (output.get().size() != 1 || !output.get()[0].isTensor())
      return false;
    auto tensor = output.get()[0].toTensor();
    if (tensor.scalar_type() != executorch::aten::ScalarType::Float ||
        tensor.dim() != 2 || tensor.size(0) != 1 || tensor.size(1) != 128000)
      return false;
    const float *values = tensor.const_data_ptr<float>();
    logits.assign(values, values + tensor.numel());
    return std::all_of(logits.begin(), logits.end(),
                       [](float v) { return std::isfinite(v); });
  };
  auto started = std::chrono::steady_clock::now();
  for (size_t pos = 0; pos < prompt.size(); pos += chunk_size) {
    int count = std::min(chunk_size, prompt.size() - pos);
    if (!forward(prompt.data() + pos, count, pos))
      return 5;
    std::printf("PREFILL pos=%zu length=%d\n", pos, count);
    std::fflush(stdout);
  }
  const double prefill_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count();
  std::ofstream dump(std::string(argv[3]) + ".f32", std::ios::binary);
  dump.write(reinterpret_cast<const char *>(logits.data()),
             logits.size() * sizeof(float));
  dump.close();
  std::ofstream tokens(std::string(argv[3]) + ".tokens");
  int generated = 0;
  int64_t pos = prompt.size();
  started = std::chrono::steady_clock::now();
  for (int i = 0; i < steps; ++i) {
    int64_t token =
        std::max_element(logits.begin(), logits.end()) - logits.begin();
    std::printf("TOKEN %d %lld\n", i, (long long)token);
    std::fflush(stdout);
    tokens << token << '\n';
    if (token == 124900)
      break;
    if (!forward(&token, 1, pos++))
      return 6;
    ++generated;
  }
  const double decode_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count();
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  std::printf("RESULT prompt_tokens=%zu prefill_ms=%.3f generated=%d "
              "decode_ms=%.3f tokens_per_second=%.3f peak_rss_mib=%.1f\n",
              prompt.size(), prefill_ms, generated, decode_ms,
              1000.0 * generated / decode_ms, usage.ru_maxrss / 1024.0);
  return 0;
}
