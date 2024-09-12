#pragma once
#ifndef clfft_1d
#define clfft_1d

int clfft_with_N(size_t N);

int clfft_R2HP_FFT(size_t N = 16);

int clfft_HP2R_IFFT(size_t N = 16);

#endif