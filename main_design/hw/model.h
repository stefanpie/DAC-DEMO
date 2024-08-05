#pragma once

#define __FLOATING_POINT_MODEL__ 0
#define __MEASURE_RUNTIME__ 0
#define __PRINT__ 1

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#if __FLOATING_POINT_MODEL__
    #pragma message("Floating point model")
    #include <cmath>
typedef float F_TYPE;
typedef float W_TYPE;

    #define m_sqrt(x) (std::sqrt(x))
    #define m_erf(x) (std::erf(x))
    #define m_tanh(x) (std::tanh(x))
    #define m_pow(x, y) (std::pow(x, y))
    #define m_exp(x) (std::exp(x))
    #define m_log(x) (std::log(x))
    #define m_abs(x) (std::abs(x))
    #define m_sin(x) (std::sin(x))
    #define m_cos(x) (std::cos(x))
    #define m_pi() ((float)3.14159265358979323846)
#else
    #pragma message("Fixed point model")
    #include "ap_fixed.h"
    #include "hls_math.h"

    #define FIXED_TYPE_F ap_fixed<32, 16>
    #define FIXED_TYPE_W ap_fixed<32, 16>
typedef FIXED_TYPE_F F_TYPE;
typedef FIXED_TYPE_W W_TYPE;

    #define m_sqrt(x) (hls::sqrt(x))
    #define m_erf(x) (hls::erf(x))
    #define m_tanh(x) (hls::tanh(x))
    #define m_pow(x, y) (hls::pow(x, y))
    #define m_exp(x) (hls::exp(x))
    #define m_log(x) (hls::log(x))
    #define m_abs(x) (hls::abs(x))
    #define m_sin(x) (hls::sin(x))
    #define m_cos(x) (hls::cos(x))
    #define m_pi() ((W_TYPE)3.14159265358979323846)
#endif

#if __PRINT__
    #define PRINT(x) x
#else
    #define PRINT(x)
#endif

#include "../../inr_hw_lib/inr_hw_lib.h"

#define INPUT_DIM 2
#define HIDDEN_DIM 256
#define OUTPUT_DIM 3
#define NUM_HIDDEN_LAYERS 4


extern "C" {
void siren_test_model(
    F_TYPE x_in[INPUT_DIM],
    F_TYPE x_out[OUTPUT_DIM],
    int load_parameters_flag,
    F_TYPE input_layer_weight_in[HIDDEN_DIM][INPUT_DIM],
    F_TYPE input_layer_bias_in[HIDDEN_DIM],
    F_TYPE input_layer_w0_in[1],
    F_TYPE hidden_layers_weight_in[NUM_HIDDEN_LAYERS][HIDDEN_DIM][HIDDEN_DIM],
    F_TYPE hidden_layers_bias_in[NUM_HIDDEN_LAYERS][HIDDEN_DIM],
    F_TYPE hidden_layers_w0_in[NUM_HIDDEN_LAYERS],
    F_TYPE output_layer_weight_in[OUTPUT_DIM][HIDDEN_DIM],
    F_TYPE output_layer_bias_in[OUTPUT_DIM]);
}