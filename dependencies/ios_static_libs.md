## React Native Executorch for iOS requires static libraries from two sources: ExecuTorch and tokenizers-cpp

### Executorch
To create all needed static libraries with executorch one has to follow [Building from source tutorial](https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html). After the executorch environment is setted up run the following command in executorch root:

```
./scripts/build_apple_frameworks.sh --Release --coreml --custom --mps --optimized --portable --quantized --xnnpack
```

It will create xcframework directories in executorch/cmake-out. Each directory contains 3 folders, for ios, simulator and mac. We are only interested in simulator and ios so enter those directories and you will find your static library (.a).

 Now from each xcframework copy static library for ios device and simulator and paste them in /packages/react-native-executorch/ios/libs/executorch. Those .a files are then linked in .podspec. Make sure that static libraries files have the same name as previous one.

 Right now we are using our ExecuTorch fork, however it will probably change in future but the steps should be the same for all Executorch repositories.

 ### tokenizers-cpp

 It's not that simple with binaries for tokenizers, however I will provide a hacky way for it now.

 First of all, checkout release/0.4 branch of react native executorch and open ExecutorchLib.xcodeproj with xcode. Make sure that the submodules are synced and updated. 

 Now in xcode, select any ios device and click run, if build is successful go to Product -> Show build Folder. Then Intermediates.noindex -> ExecutorchLib.build -> Debug-iphoneos -> ExecutorchLib.build -> cmake -> build -> tokenizers-cpp. From here copy libtokenizer_c.a and libtokenizer_cpp.a, also you will need to copy sentencepiece/src/libsentecepiece.a. Paste those 3 files /packages/react-native-executorch/ios/libs/tokenizers-cpp/physical-arm64-release.

 You have to do the same for simulator.