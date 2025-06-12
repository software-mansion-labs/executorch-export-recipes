# Bumping Executorch version

## Kotlin
Build the Executorch archive from source with the JNI changes (https://github.com/software-mansion-labs/executorch). More information on how to build: https://pytorch.org/executorch/0.6/using-executorch-android.html. Replace the archive in third-party/ (by default the archive is loaded from jitpack which will change after pushing on main -- to see the changes earlier change build.gradle to load the dependency from a file).
## Objective C
Build xcframeworks from source (https://pytorch.org/executorch/0.6/using-executorch-ios.html). Replace the relevant frameworks in third-party/ios/ExecutorchLib. Build ExecutorchLib using ExecutorchLib/build.sh and replace the resulting framework in ios/.
## C++ (Android)
On the archive generated for Kotlin dependency (for C++ the JNI changes don't need to be present):\
`unzip executorch.aar` and replace the .so files inside the unziped archive for both architectures in third-party/android/libs.
## C++ (iOS)
Gather all the .a files that are stored inside the framework folders used by Objective C and replace them in ios/libs.
## Headers
Include headers for executorch are generated during build. You can either copy them from the executorch repo (the folder `include` will be present only after building), or copy them from the executorch.xcframework/ios-arm64/Headers. To bump the header version, replace the folder in third-party/include.
