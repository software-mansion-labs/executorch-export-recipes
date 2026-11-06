#!/bin/zsh
# Gradle-free release APK of the Echo demo: NDK -> libecho_jni.so; javac -> d8 -> classes.dex; aapt2 link;
# add the runtime libs; zipalign; sign with a local debug key (created on first run).
#   RNE_LIBS    react-native-executorch v0.11.1-libs, every *-android-arm64-v8a.tar.gz and headers.tar.gz
#               extracted here (ExecuTorch 1.4.1 fork build: core, XNNPACK, Vulkan with fork #27, QNN)
#   QAIRT_ROOT  QAIRT 2.47 SDK (the version the QNN .pte files were exported with)
#   FBJNI_SO    arm64 libfbjni.so (com.facebook.fbjni:fbjni AAR, jni/arm64-v8a/), needed by libexecutorch.so
#   ANDROID_SDK (default ~/Library/Android/sdk), NDK_VERSION (default 27.1.12297006)
set -e
APP=${0:a:h}; OUT=$APP/build
: ${RNE_LIBS:?set RNE_LIBS} ${QAIRT_ROOT:?set QAIRT_ROOT} ${FBJNI_SO:?set FBJNI_SO}
SDK=${ANDROID_SDK:-$HOME/Library/Android/sdk}; BT=$SDK/build-tools/35.0.0
NDK=$(echo $SDK/ndk/${NDK_VERSION:-27.1.12297006}/toolchains/llvm/prebuilt/*)
CXX=$NDK/bin/aarch64-linux-android26-clang++
JAR=$SDK/platforms/android-34/android.jar
LIB=$OUT/lib/arm64-v8a
rm -rf $OUT; mkdir -p $LIB $OUT/classes
so() { find -L $RNE_LIBS -path '*arm64-v8a*' -name $1 | head -1; }
cp $(so libexecutorch.so) $LIB/libexecutorch_jni.so   # its SONAME
for l in libxnnpack_executorch_backend libvulkan_executorch_backend libqnn_executorch_backend; do cp $(so $l.so) $LIB/; done
cp $NDK/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $FBJNI_SO $LIB/
cp $QAIRT_ROOT/lib/aarch64-android/{libQnnHtp,libQnnSystem,libQnnHtpV81Stub}.so $QAIRT_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so $LIB/
INC=$(find -L $RNE_LIBS -type d -name include -path '*headers*' | head -1)
$CXX -std=c++17 -O3 -DNDEBUG -shared -fPIC -I$INC -I$INC/executorch/extension/llm/tokenizers/include \
  -o $LIB/libecho_jni.so $APP/cpp/echo_jni.cpp $APP/cpp/echo_engine.cpp \
  -L$LIB -l:libexecutorch_jni.so -Wl,--no-as-needed -lxnnpack_executorch_backend -lvulkan_executorch_backend \
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
