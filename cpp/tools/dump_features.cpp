// Host-compilable frontend parity dump: fbank of the first 10 s window.
//   g++ -O2 tools/dump_features.cpp third_party/kissfft/kiss_fft.c \
//       third_party/kissfft/kiss_fftr.c -I. -Isrc -Ithird_party/kissfft \
//       -o dump_features && ./dump_features model_dir wav out.bin
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "fbank.hpp"
#include "params.hpp"
#include "wav_reader.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: dump_features model_dir wav out.bin\n");
        return 1;
    }
    Params P;
    P.Load(argv[1]);
    WavData wav = read_wav(argv[2]);
    int n = std::min((int)wav.samples.size(), 160000);
    std::vector<float> fbank(998 * 80);
    int m = kaldi_fbank(P, wav.samples.data(), n, fbank.data());
    printf("frames %d\n", m);
    FILE* f = fopen(argv[3], "wb");
    fwrite(fbank.data(), 4, (size_t)m * 80, f);
    fclose(f);
    return 0;
}
