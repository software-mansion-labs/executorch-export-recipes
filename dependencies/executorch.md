# Kotlin dependency
Build the Executorch archive from source with the JNI changes (https://github.com/software-mansion-labs/executorch). More information on how to build: https://pytorch.org/executorch/0.6/using-executorch-android.html. Replace the archive in third-party/
# Objective C
Build xcframeworks from source (https://pytorch.org/executorch/0.6/using-executorch-ios.html). Replace the relevant frameworks in third-party/ios/ExecutorchLib. Build ExecutorchLib using ExecutorchLib/build.sh and replace the resulting framework in ios/.
# C++ (Android)
On the archive generated:\
`unzip executorch.aar -d .` and replace the .so files inside the unziped archive for both architectures in Android/libs.
# C++ (iOS)
Gather all the .a files that are stored inside the framework folders used by Objective C and replace them in ios/libs.