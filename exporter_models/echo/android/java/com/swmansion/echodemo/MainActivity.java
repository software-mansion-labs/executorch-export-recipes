package com.swmansion.echodemo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
  private static final int RATE = 24000;
  private static final int MAX_SECONDS = 10;

  // backbone file, fallback for texts that do not fit the QNN bucket, label, codec decoder ("" = XNNPACK),
  // its static frame count (qnn/export_codec_qnn.py: 10 s; longer clips use the XNNPACK decoder)
  private static final String[][] BACKBONES = {
    {"echo_qnn_fp16_t256.pte", "echo_vulkan_8da4w.pte", "NPU (Qualcomm QNN), fp16", "echo_codec_qnn_a16w8_t256.pte", "256"},
    {"echo_qnn_a16w8_p320_t256.pte", "echo_vulkan_8da4w.pte", "NPU (Qualcomm QNN), int8", "echo_codec_qnn_a16w8_t256.pte", "256"},
    {"echo_vulkan_8da4w.pte", "", "GPU (Vulkan), int4", "", "0"},
    {"echo_vulkan_fp16.pte", "", "GPU (Vulkan), fp16", "", "0"},
  };
  // language code, label, sentence to read aloud, default text to speak
  private static final String[][] LANGS = {
    {"en", "English", "The quick brown fox jumps over the lazy dog, and then it runs back home before the rain starts.",
        "Hi! This voice was cloned a moment ago, and everything is running right here on the phone, without any cloud."},
    {"pl", "Polski", "Wczoraj wieczorem poszedłem na długi spacer po parku i słuchałem, jak ptaki śpiewają w koronach drzew.",
        "Cześć! Ten głos został sklonowany przed chwilą, a wszystko działa bezpośrednio na telefonie, bez chmury."},
    {"de", "Deutsch", "Gestern Abend bin ich lange im Park spazieren gegangen und habe den Vögeln in den Bäumen zugehört.",
        "Hallo! Diese Stimme wurde gerade eben geklont, und alles läuft direkt auf dem Telefon, ganz ohne Cloud."},
    {"fr", "Français", "Hier soir, je me suis longuement promené dans le parc en écoutant les oiseaux chanter dans les arbres.",
        "Bonjour ! Cette voix vient d'être clonée, et tout fonctionne directement sur le téléphone, sans aucun cloud."},
    {"es", "Español", "Ayer por la tarde di un largo paseo por el parque y escuché a los pájaros cantar en los árboles.",
        "¡Hola! Esta voz se acaba de clonar, y todo funciona directamente en el teléfono, sin ninguna nube."},
    {"it", "Italiano", "Ieri sera ho fatto una lunga passeggiata nel parco ascoltando gli uccelli cantare tra gli alberi.",
        "Ciao! Questa voce è stata appena clonata, e tutto funziona direttamente sul telefono, senza alcun cloud."},
    {"pt", "Português", "Ontem à noite dei um longo passeio pelo parque e ouvi os pássaros a cantar nas árvores.",
        "Olá! Esta voz acabou de ser clonada, e tudo funciona diretamente no telemóvel, sem qualquer nuvem."},
  };

  private final ExecutorService worker = Executors.newSingleThreadExecutor();
  private final Handler ui = new Handler(Looper.getMainLooper());
  private final List<String[]> backbones = new ArrayList<>();

  private TextView status, refInfo, stats;
  private EditText refText, text;
  private Spinner backboneSpinner, langSpinner;
  private Button recordBtn, playRefBtn, generateBtn, playOutBtn;

  private volatile boolean recording;
  private float[] recorded, output;
  private boolean modelsReady, refReady;
  private String loadedBackbone;
  private AudioTrack track;

  @Override
  protected void onCreate(Bundle b) {
    super.onCreate(b);
    try {  // the Hexagon DSP loads the QNN skel library from here
      android.system.Os.setenv("ADSP_LIBRARY_PATH", getApplicationInfo().nativeLibraryDir
          + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp", true);
    } catch (android.system.ErrnoException ignored) {
    }
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    buildUi();
    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      requestPermissions(new String[] {Manifest.permission.RECORD_AUDIO}, 1);
    }
    File dir = getExternalFilesDir(null);
    for (String[] bb : BACKBONES) {
      if (new File(dir, bb[0]).exists() && (bb[1].isEmpty() || new File(dir, bb[1]).exists())
          && (bb[3].isEmpty() || new File(dir, bb[3]).exists())) backbones.add(bb);
    }
    if (backbones.isEmpty() || !new File(dir, "tokenizer.json").exists()) {
      status.setText("Models missing. Push them to " + dir);
      return;
    }
    List<String> labels = new ArrayList<>();
    for (String[] bb : backbones) labels.add(bb[2] + (bb[1].isEmpty() ? "" : ", GPU fallback for long texts"));
    backboneSpinner.setAdapter(adapter(labels));
    backboneSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
      @Override public void onItemSelected(AdapterView<?> p, View v, int pos, long id) { loadModels(backbones.get(pos)); }
      @Override public void onNothingSelected(AdapterView<?> p) {}
    });
  }

  // ------------------------------------------------------------------ UI

  private void buildUi() {
    LinearLayout col = new LinearLayout(this);
    col.setOrientation(LinearLayout.VERTICAL);
    int pad = dp(16);
    col.setPadding(pad, pad, pad, pad);

    TextView title = new TextView(this);
    title.setText("Echo: on-device voice cloning");
    title.setTextSize(22);
    title.setTypeface(Typeface.DEFAULT_BOLD);
    col.addView(title);
    status = label(col, "Loading…");

    section(col, "Model");
    backboneSpinner = new Spinner(this);
    col.addView(backboneSpinner);

    section(col, "1. Language");
    langSpinner = new Spinner(this);
    List<String> names = new ArrayList<>();
    for (String[] l : LANGS) names.add(l[1]);
    langSpinner.setAdapter(adapter(names));
    langSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
      @Override public void onItemSelected(AdapterView<?> p, View v, int pos, long id) {
        refText.setText(LANGS[pos][2]);
        text.setText(LANGS[pos][3]);
      }
      @Override public void onNothingSelected(AdapterView<?> p) {}
    });
    col.addView(langSpinner);

    section(col, "2. Record your voice reading this sentence (3 to 10 s)");
    refText = edit(col, 2);
    LinearLayout row = new LinearLayout(this);
    recordBtn = button(row, "● Record", v -> toggleRecording());
    playRefBtn = button(row, "▶ Play recording", v -> play(recorded));
    col.addView(row);
    refInfo = label(col, "No recording yet.");

    section(col, "3. Text to speak in your voice");
    text = edit(col, 4);
    LinearLayout row2 = new LinearLayout(this);
    generateBtn = button(row2, "Generate", v -> generate());
    playOutBtn = button(row2, "▶ Play again", v -> play(output));
    col.addView(row2);
    stats = label(col, "");
    stats.setTypeface(Typeface.MONOSPACE);

    ScrollView scroll = new ScrollView(this);
    scroll.addView(col);
    setContentView(scroll);
    refreshButtons();
  }

  private void section(LinearLayout col, String s) {
    TextView t = new TextView(this);
    t.setText(s);
    t.setTextSize(16);
    t.setTypeface(Typeface.DEFAULT_BOLD);
    t.setPadding(0, dp(18), 0, dp(4));
    col.addView(t);
  }

  private TextView label(LinearLayout col, String s) {
    TextView t = new TextView(this);
    t.setText(s);
    t.setPadding(0, dp(4), 0, dp(4));
    col.addView(t);
    return t;
  }

  private EditText edit(LinearLayout col, int lines) {
    EditText e = new EditText(this);
    e.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
    e.setMinLines(lines);
    e.setGravity(Gravity.TOP);
    col.addView(e);
    return e;
  }

  private Button button(LinearLayout row, String s, View.OnClickListener l) {
    Button btn = new Button(this);
    btn.setText(s);
    btn.setAllCaps(false);
    btn.setOnClickListener(l);
    row.addView(btn, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
    return btn;
  }

  private ArrayAdapter<String> adapter(List<String> items) {
    return new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, items) {
      @Override public View getView(int pos, View convert, android.view.ViewGroup parent) {
        return row(items.get(pos));
      }
      @Override public View getDropDownView(int pos, View convert, android.view.ViewGroup parent) {
        return row(items.get(pos));
      }
    };
  }

  private TextView row(String s) {
    TextView t = new TextView(this);
    t.setText(s);
    t.setTextSize(17);
    t.setTextColor(0xFFFFFFFF);
    t.setBackgroundColor(0xFF2A2A2E);
    t.setPadding(dp(12), dp(12), dp(12), dp(12));
    return t;
  }

  private int dp(int v) {
    return (int) (v * getResources().getDisplayMetrics().density);
  }

  private void refreshButtons() {
    recordBtn.setEnabled(modelsReady || recording);
    playRefBtn.setEnabled(recorded != null && !recording);
    generateBtn.setEnabled(modelsReady && refReady && !recording);
    playOutBtn.setEnabled(output != null);
  }

  private void busy(boolean b) {
    modelsReady = !b && loadedBackbone != null;
    backboneSpinner.setEnabled(!b);
    refreshButtons();
  }

  // ------------------------------------------------------------------ models

  private void loadModels(String[] bb) {
    final String backbone = bb[0];
    if (backbone.equals(loadedBackbone)) return;
    status.setText("Loading models (" + bb[2] + ")…");
    loadedBackbone = null;
    busy(true);
    final float[] ref = recorded;
    worker.execute(() -> {
      long t0 = System.nanoTime();
      String err = Echo.load(getExternalFilesDir(null).getAbsolutePath(), backbone, bb[1], bb[3], Integer.parseInt(bb[4]));
      String refErr = err == null && ref != null ? Echo.setReference(ref) : null;
      long ms = (System.nanoTime() - t0) / 1000000;
      ui.post(() -> {
        if (err != null) {
          status.setText("Load failed: " + err);
        } else {
          loadedBackbone = backbone;
          status.setText("Ready. Models loaded in " + ms + " ms.");
          refReady = ref != null && refErr == null;
        }
        busy(false);
      });
    });
  }

  // ------------------------------------------------------------------ recording

  private void toggleRecording() {
    if (recording) {
      recording = false;
      return;
    }
    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      requestPermissions(new String[] {Manifest.permission.RECORD_AUDIO}, 1);
      return;
    }
    recording = true;
    recordBtn.setText("■ Stop");
    refreshButtons();
    new Thread(this::recordLoop).start();
  }

  private void recordLoop() {
    int rate = RATE;
    int min = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
    boolean decimate = false;
    if (min <= 0) {  // no native 24 kHz: record 48 kHz and halve
      rate = 48000;
      decimate = true;
      min = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
    }
    AudioRecord rec;
    try {
      rec = new AudioRecord(MediaRecorder.AudioSource.MIC, rate, AudioFormat.CHANNEL_IN_MONO,
          AudioFormat.ENCODING_PCM_16BIT, Math.max(min, rate));
    } catch (SecurityException e) {
      ui.post(() -> { refInfo.setText("Microphone permission denied."); stopUi(); });
      return;
    }
    short[] all = new short[rate * (MAX_SECONDS + 1)];
    int n = 0;
    rec.startRecording();
    long start = System.nanoTime();
    final int r = rate;
    while (recording && n < rate * MAX_SECONDS) {
      int got = rec.read(all, n, Math.min(rate / 10, rate * MAX_SECONDS - n));
      if (got > 0) n += got;
      final double sec = (double) n / r;
      ui.post(() -> refInfo.setText(String.format("Recording… %.1f s (stops at %d s)", sec, MAX_SECONDS)));
    }
    rec.stop();
    rec.release();
    recording = false;
    float[] wav = new float[decimate ? n / 2 : n];
    float peak = 1e-6f;
    for (int i = 0; i < wav.length; i++) {
      wav[i] = decimate ? (all[2 * i] + all[2 * i + 1]) / 65536f : all[i] / 32768f;
      peak = Math.max(peak, Math.abs(wav[i]));
    }
    float gain = Math.min(0.9f / peak, 10f);  // normalize quiet recordings
    for (int i = 0; i < wav.length; i++) wav[i] *= gain;
    ui.post(() -> {
      stopUi();
      recorded = wav;
      encodeReference();
    });
  }

  private void stopUi() {
    recording = false;
    recordBtn.setText("● Record");
    refreshButtons();
  }

  private void encodeReference() {
    refReady = false;
    refInfo.setText(String.format("Recorded %.1f s. Encoding your voice…", recorded.length / (double) RATE));
    busy(true);
    final float[] wav = recorded;
    worker.execute(() -> {
      String err = Echo.setReference(wav);
      String info = err == null ? Echo.refStats() : null;
      ui.post(() -> {
        refReady = err == null;
        refInfo.setText(err == null ? "Voice captured: " + info : "Could not use the recording: " + err);
        busy(false);
      });
    });
  }

  // ------------------------------------------------------------------ generation

  private void generate() {
    final String lang = LANGS[langSpinner.getSelectedItemPosition()][0];
    final String ref = refText.getText().toString().trim();
    final String txt = text.getText().toString().trim();
    if (txt.isEmpty()) return;
    busy(true);
    stats.setText("");
    status.setText("Generating…");
    worker.execute(() -> {
      int frames = Echo.estimateFrames(txt, ref, Echo.speechFrames());
      long t0 = System.nanoTime();
      Echo.Result r = Echo.generate(ref, txt, lang, frames, System.nanoTime(),
          msg -> ui.post(() -> status.setText("Generating… " + msg)));
      double total = (System.nanoTime() - t0) / 1e6;
      ui.post(() -> {
        busy(false);
        if (r.error != null) {
          status.setText("Failed: " + r.error);
          return;
        }
        output = r.audio;
        double secs = r.audio.length / (double) RATE;
        status.setText(String.format("Done: %.1f s of audio in %.1f s (%.1fx real time).", secs, total / 1000,
            secs / (total / 1000)));
        stats.setText(r.stats + String.format("\ntotal: %.0f ms", total));
        saveWav(r.audio);
        refreshButtons();
        play(r.audio);
      });
    });
  }

  private void play(float[] audio) {
    if (audio == null) return;
    if (track != null) { track.release(); track = null; }
    track = new AudioTrack.Builder()
        .setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
        .setAudioFormat(new AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(RATE).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
        .setTransferMode(AudioTrack.MODE_STATIC)
        .setBufferSizeInBytes(audio.length * 4)
        .build();
    track.write(audio, 0, audio.length, AudioTrack.WRITE_BLOCKING);
    track.play();
  }

  private void saveWav(float[] audio) {
    File f = new File(getExternalFilesDir(null), "last_output.wav");
    ByteBuffer bb = ByteBuffer.allocate(44 + audio.length * 2).order(ByteOrder.LITTLE_ENDIAN);
    bb.put("RIFF".getBytes()).putInt(36 + audio.length * 2).put("WAVEfmt ".getBytes()).putInt(16)
        .putShort((short) 1).putShort((short) 1).putInt(RATE).putInt(RATE * 2).putShort((short) 2)
        .putShort((short) 16).put("data".getBytes()).putInt(audio.length * 2);
    for (float v : audio) bb.putShort((short) Math.max(-32768, Math.min(32767, Math.round(v * 32767))));
    try (FileOutputStream o = new FileOutputStream(f)) {
      o.write(bb.array());
    } catch (IOException ignored) {
    }
  }
}
