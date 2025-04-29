# Dependencies outside Executorch
## OpenCV
### iOS
A minimal build for iOS is hosted as a custom cocoapod here: https://github.com/software-mansion-labs/opencv-rne. The minimal build originates from https://github.com/nihui/opencv-mobile. To bump, change the version in the repo and deploy a new pod version.
### Android
A minimal build from the same source is stored in third-party/android/. To bump change the binaries in libs/ and change the headers in include/ (opencv2/ **and** c10/).

## Ada
Ada is a C++ header-only library for parsing urls (https://github.com/ada-url/ada). To bump change the headers in common/. 