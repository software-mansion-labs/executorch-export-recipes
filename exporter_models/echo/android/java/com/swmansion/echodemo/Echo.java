package com.swmansion.echodemo;

/** JNI bridge to the C++ engine (cpp/echo_engine.cpp). One engine per process; calls are serialized. */
final class Echo {
  static {
    System.loadLibrary("echo_jni");
  }

  interface Listener {
    void onProgress(String message);
  }

  static final class Result {
    final float[] audio;
    final String error;
    final String stats;

    Result(float[] audio, String error, String stats) {
      this.audio = audio;
      this.error = error;
      this.stats = stats;
    }
  }

  /**
   * decoder: codec decoder file ("" for the dynamic XNNPACK one); decoderFrames: its static frame
   * count (0 = dynamic). Returns null on success, else an error message.
   */
  static native String load(String dir, String backbone, String fallback, String decoder, int decoderFrames);

  /** wav: mono 24 kHz. Returns null on success, else an error message. */
  static native String setReference(float[] wav);

  static native int speechFrames();

  static native String refStats();

  static native Result generate(String refText, String text, String lang, int frames, long seed, Listener listener);

  /** The notebook's estimate_frames: the text's spoken weight at the reference's rate. */
  static int estimateFrames(String text, String refText, int refFrames) {
    double frames = weight(text) / (weight(refText) / refFrames);
    if (frames < 50) frames = 50 * Math.cbrt(frames / 50);
    return Math.max(1, (int) frames);
  }

  private static double weight(String s) {
    double w = 0;
    for (int i = 0; i < s.length(); ) {
      int cp = s.codePointAt(i);
      i += Character.charCount(cp);
      switch (Character.getType(cp)) {
        case Character.SPACE_SEPARATOR: case Character.LINE_SEPARATOR: case Character.PARAGRAPH_SEPARATOR:
          w += 0.2; break;
        case Character.NON_SPACING_MARK: case Character.ENCLOSING_MARK: case Character.COMBINING_SPACING_MARK:
          break;
        case Character.CONNECTOR_PUNCTUATION: case Character.DASH_PUNCTUATION: case Character.START_PUNCTUATION:
        case Character.END_PUNCTUATION: case Character.INITIAL_QUOTE_PUNCTUATION: case Character.FINAL_QUOTE_PUNCTUATION:
        case Character.OTHER_PUNCTUATION: case Character.MATH_SYMBOL: case Character.CURRENCY_SYMBOL:
        case Character.MODIFIER_SYMBOL: case Character.OTHER_SYMBOL:
          w += 0.5; break;
        case Character.DECIMAL_DIGIT_NUMBER: case Character.LETTER_NUMBER: case Character.OTHER_NUMBER:
          w += 3.5; break;
        default:
          w += 1.0;
      }
    }
    return w;
  }
}
