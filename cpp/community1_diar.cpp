// pyannote community-1 speaker diarization on AX650N (C++).
//
//   community1_diar --model-dir models --wav in.wav --out out.rttm
//                   [--step 2.0] [--num-speakers 0]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "ax_sys_api.h"

#include "diar_engine.hpp"
#include "wav_reader.hpp"

static void usage() {
    fprintf(stderr,
            "usage: community1_diar --model-dir DIR --wav WAV [--out RTTM] "
            "[--step 2.0] [--num-speakers 0]\n");
    exit(1);
}

int main(int argc, char** argv) {
    std::string model_dir = "models", wav_path, out_path = "output.rttm";
    float step = 2.0f;
    int num_speakers = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--model-dir") model_dir = val();
        else if (a == "--wav") wav_path = val();
        else if (a == "--out") out_path = val();
        else if (a == "--step") step = (float)atof(val().c_str());
        else if (a == "--num-speakers") num_speakers = atoi(val().c_str());
        else usage();
    }
    if (wav_path.empty()) usage();

    int ret = AX_SYS_Init();
    if (ret != 0) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    c1::DiarEngine engine;
    if (!engine.Init(model_dir)) {
        fprintf(stderr, "engine init failed\n");
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double load_s = std::chrono::duration<double>(t1 - t0).count();

    WavData wav = read_wav(wav_path);
    if (wav.sample_rate != 16000)
        fprintf(stderr, "warning: input is %d Hz, model expects 16 kHz\n", wav.sample_rate);

    std::vector<c1::Segment> rttm;
    engine.Run(wav, step, num_speakers, rttm);
    auto t2 = std::chrono::steady_clock::now();
    double infer_s = std::chrono::duration<double>(t2 - t1).count();
    double dur = wav.samples.size() / 16000.0;

    std::string uri = wav_path.substr(wav_path.find_last_of('/') + 1);
    size_t dot = uri.find_last_of('.');
    if (dot != std::string::npos) uri = uri.substr(0, dot);
    FILE* f = fopen(out_path.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    for (auto& s : rttm)
        fprintf(f, "SPEAKER %s 1 %7.3f %7.3f <NA> <NA> SPEAKER_%02d <NA> <NA>\n",
                uri.c_str(), s.start, s.dur, s.spk);
    fclose(f);

    printf("load %.1f s | inference %.1f s | audio %.1f s | RTF %.4f | %zu segments\n",
           load_s, infer_s, dur, infer_s / dur, rttm.size());
    return 0;
}
