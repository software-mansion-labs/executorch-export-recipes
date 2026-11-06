#include <jni.h>

#include <mutex>

#include "echo_engine.h"

namespace {
echo::Engine* g_engine = nullptr;
std::mutex g_mu;

std::string str(JNIEnv* env, jstring s) {
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string r(c);
  env->ReleaseStringUTFChars(s, c);
  return r;
}
}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_com_swmansion_echodemo_Echo_load(JNIEnv* env, jclass, jstring dir, jstring backbone,
                                                                jstring fallback, jstring decoder, jint decoderFrames) {
  std::lock_guard<std::mutex> l(g_mu);
  delete g_engine;
  g_engine = new echo::Engine();
  const std::string dec = decoder ? str(env, decoder) : "";
  std::string e = g_engine->load(str(env, dir), str(env, backbone), fallback ? str(env, fallback) : "",
                                 dec.empty() ? echo::kDynamicDecoder : dec, echo::kEncoder, decoderFrames);
  if (!e.empty()) {
    delete g_engine;
    g_engine = nullptr;
  }
  return e.empty() ? nullptr : env->NewStringUTF(e.c_str());
}

// Returns the trimmed audio that was encoded (empty array + error via lastError on failure).
JNIEXPORT jstring JNICALL Java_com_swmansion_echodemo_Echo_setReference(JNIEnv* env, jclass, jfloatArray wav) {
  std::lock_guard<std::mutex> l(g_mu);
  if (!g_engine) return env->NewStringUTF("models not loaded");
  std::vector<float> w(env->GetArrayLength(wav));
  env->GetFloatArrayRegion(wav, 0, (jsize)w.size(), w.data());
  std::string e = g_engine->set_reference(w);
  return e.empty() ? nullptr : env->NewStringUTF(e.c_str());
}

JNIEXPORT jobject JNICALL Java_com_swmansion_echodemo_Echo_generate(JNIEnv* env, jclass, jstring refText, jstring text,
                                                                    jstring lang, jint frames, jlong seed,
                                                                    jobject listener) {
  std::lock_guard<std::mutex> l(g_mu);
  jclass rc = env->FindClass("com/swmansion/echodemo/Echo$Result");
  jmethodID ctor = env->GetMethodID(rc, "<init>", "([FLjava/lang/String;Ljava/lang/String;)V");
  if (!g_engine) return env->NewObject(rc, ctor, nullptr, env->NewStringUTF("models not loaded"), nullptr);
  jclass lc = env->GetObjectClass(listener);
  jmethodID onProgress = env->GetMethodID(lc, "onProgress", "(Ljava/lang/String;)V");
  std::vector<float> audio;
  std::string e = g_engine->generate(str(env, refText), str(env, text), str(env, lang), frames, (uint64_t)seed, &audio,
                                     [&](const std::string& s) {
                                       jstring js = env->NewStringUTF(s.c_str());
                                       env->CallVoidMethod(listener, onProgress, js);
                                       env->DeleteLocalRef(js);
                                     });
  if (!e.empty()) return env->NewObject(rc, ctor, nullptr, env->NewStringUTF(e.c_str()), nullptr);
  jfloatArray out = env->NewFloatArray((jsize)audio.size());
  env->SetFloatArrayRegion(out, 0, (jsize)audio.size(), audio.data());
  return env->NewObject(rc, ctor, out, nullptr, env->NewStringUTF(g_engine->stats().to_string().c_str()));
}

JNIEXPORT jint JNICALL Java_com_swmansion_echodemo_Echo_speechFrames(JNIEnv*, jclass) {
  std::lock_guard<std::mutex> l(g_mu);
  return g_engine ? g_engine->speech_frames() : 0;
}

JNIEXPORT jstring JNICALL Java_com_swmansion_echodemo_Echo_refStats(JNIEnv* env, jclass) {
  std::lock_guard<std::mutex> l(g_mu);
  if (!g_engine) return nullptr;
  char b[96];
  std::snprintf(b, sizeof b, "%.1f s of speech, encoded as %.0f s in %.0f ms", g_engine->speech_frames() / 25.0,
                g_engine->ref_frames() / 25.0, g_engine->stats().encode_ref_ms);
  return env->NewStringUTF(b);
}
}

// Our own JNI_OnLoad: without it, ART's dlsym finds libexecutorch_jni.so's, which registers the
// org.pytorch.executorch Java classes this app does not ship.
extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM*, void*) { return JNI_VERSION_1_6; }
