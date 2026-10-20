#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
using namespace executorch::extension;
using namespace executorch::runtime;
int main(int argc,char**argv){
 if(argc<5){fprintf(stderr,"model input.f32 side output.f32 [runs]\n");return 2;}
 runtime_init();
 int side=atoi(argv[3]),runs=argc>5?atoi(argv[5]):3;
 std::vector<float> pixels(3*side*side);
 std::ifstream file(argv[2],std::ios::binary);file.read((char*)pixels.data(),pixels.size()*4);
 if(!file){fprintf(stderr,"invalid input\n");return 3;}
 auto input=make_tensor_ptr({1,3,side,side},pixels.data(),executorch::aten::ScalarType::Float);
 Module model(argv[1],Module::LoadMode::Mmap);
 for(int run=0;run<runs;++run){
  auto start=std::chrono::steady_clock::now();
  auto output=model.execute("forward",std::vector<EValue>{*input});
  if(!output.ok()){fprintf(stderr,"execute error %x\n",unsigned(output.error()));return 4;}
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  if(output.get().size()!=1||!output.get()[0].isTensor()){fprintf(stderr,"invalid output\n");return 5;}
  auto tensor=output.get()[0].toTensor();
  if(tensor.scalar_type()!=executorch::aten::ScalarType::Float||tensor.numel()!=side*side){fprintf(stderr,"wrong output contract\n");return 6;}
  const float* data=tensor.const_data_ptr<float>();
  int bad=0;float lo=data[0],hi=data[0];
  for(size_t i=0;i<tensor.numel();++i){bad+=!std::isfinite(data[i]);lo=std::min(lo,data[i]);hi=std::max(hi,data[i]);}
  printf("run %d ms %.3f nonfinite %d range %.8g %.8g\n",run,ms,bad,lo,hi);fflush(stdout);
  if(bad)return 7;
  std::ofstream dump(argv[4],std::ios::binary);dump.write((const char*)data,tensor.nbytes());
 }
}
