#!/bin/zsh
# Gradle-free release APK of the Echo demo: NDK -> libecho_jni.so; javac -> d8 -> classes.dex; aapt2 link;
# add the runtime libs; zipalign; sign with a local debug key (created on first run).
#   ET_DIR   ExecuTorch Android build: include/ (headers) and lib/ with libexecutorch.so, the XNNPACK, Vulkan
#            and QNN backend libs, libc++_shared.so, libfbjni.so
#            (Vulkan from a runtime with fork #27, the dq8ca M > K fix, or 8da4w breaks on P > 512 prompts)
#   QNN_LIBS QAIRT 2.47 runtime for the device's HTP (libQnnHtp.so, libQnnSystem.so, libQnnHtpV81Stub/Skel.so)
#   ANDROID_SDK (default ~/Library/Android/sdk), NDK_VERSION (default 27.1.12297006)
set -e
APP=${0:a:h}; OUT=$APP/build
: ${ET_DIR:?set ET_DIR} ${QNN_LIBS:?set QNN_LIBS}
SDK=${ANDROID_SDK:-$HOME/Library/Android/sdk}; BT=$SDK/build-tools/35.0.0
NDK=$SDK/ndk/${NDK_VERSION:-27.1.12297006}
CXX=$(echo $NDK/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android26-clang++)
JAR=$SDK/platforms/android-34/android.jar
rm -rf $OUT; mkdir -p $OUT/lib/arm64-v8a $OUT/classes
cp $ET_DIR/lib/libexecutorch.so $OUT/lib/arm64-v8a/libexecutorch_jni.so   # its SONAME
cp $ET_DIR/lib/{libxnnpack_executorch_backend,libvulkan_executorch_backend,libqnn_executorch_backend,libc++_shared,libfbjni}.so \
  $OUT/lib/arm64-v8a/
cp $QNN_LIBS/*.so $OUT/lib/arm64-v8a/
$CXX -std=c++17 -O3 -DNDEBUG -shared -fPIC -I$ET_DIR/include -I$ET_DIR/include/executorch/extension/llm/tokenizers/include \
  -o $OUT/lib/arm64-v8a/libecho_jni.so $APP/cpp/echo_jni.cpp $APP/cpp/echo_engine.cpp \
  -L$OUT/lib/arm64-v8a -l:libexecutorch_jni.so -Wl,--no-as-needed -lxnnpack_executorch_backend -lvulkan_executorch_backend \
  -lqnn_executorch_backend -Wl,--as-needed -lc++_shared -llog
JAVA_HOME=$(/usr/libexec/java_home -v 17); export JAVA_HOME
$JAVA_HOME/bin/javac --release 8 -classpath $JAR -d $OUT/classes $(find $APP/java -name '*.java')
$BT/d8 --min-api 26 --lib $JAR --output $OUT $(find $OUT/classes -name '*.class')
$BT/aapt2 link -o $OUT/base.apk -I $JAR --manifest $APP/AndroidManifest.xml
cd $OUT && zip -q base.apk classes.dex && zip -q -r base.apk lib
$BT/zipalign -f -p 4 base.apk aligned.apk
KS=$APP/debug.keystore
[ -f $KS ] || $JAVA_HOME/bin/keytool -genkeypair -keystore $KS -storepass android -keypass android \
  -alias androiddebugkey -keyalg RSA -validity 10000 -dname "CN=Android Debug,O=Android,C=US"
$BT/apksigner sign --ks $KS --ks-pass pass:android --key-pass pass:android --ks-key-alias androiddebugkey \
  --out echo-demo-release.apk aligned.apk
ls -la $OUT/echo-demo-release.apk
