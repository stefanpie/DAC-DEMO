#pragma once

// Array Helpers //

#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_vector.h"
#include <iostream>
#include <math.h>

template <const int M, typename T_in, typename T_out>
void cast_1d(T_in in[M], T_out out[M]) {
    for (int i = 0; i < M; i++) {
        out[i] = (T_out)in[i];
    }
}

template <const int M, const int N, typename T_in, typename T_out>
void cast_2d(T_in in[M][N], T_out out[M][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i][j] = (T_out)in[i][j];
        }
    }
}

template <const int M, const int N, const int O, typename T_in, typename T_out>
void cast_3d(T_in in[M][N][O], T_out out[M][N][O]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                out[i][j][k] = (T_out)in[i][j][k];
            }
        }
    }
}

template <const int M, typename T>
void copy_1d(T in[M], T out[M]) {
    for (int i = 0; i < M; i++) {
        out[i] = in[i];
    }
}

template <const int M, const int N, typename T>
void copy_2d(T in[M][N], T out[M][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i][j] = in[i][j];
        }
    }
}

template <const int M, const int N, const int O, typename T>
void copy_3d(T in[M][N][O], T out[M][N][O]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                out[i][j][k] = in[i][j][k];
            }
        }
    }
}

template <const int M, typename T = float>
void load_data_1d(const char *fp, T arr[M]) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), M, f);
    fclose(f);
}

template <const int M, const int N, typename T = float>
void load_data_2d(const char *fp, T arr[M][N]) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), M * N, f);
    fclose(f);
}

template <const int M, const int N, const int O, typename T = float>
void load_data_3d(const char *fp, T arr[M][N][O]) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), M * N * O, f);
    fclose(f);
}

template <const int M, typename T = float>
void load_data_var_1d(const char *fp, T arr[M], int i) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), i, f);
    fclose(f);
}

template <const int M, const int N, typename T = float>
void load_data_var_2d(const char *fp, T arr[M][N], int i, int j) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), i * j, f);
    fclose(f);
}

template <const int M, const int N, const int O, typename T = float>
void load_data_var_3d(const char *fp, T arr[M][N][O], int i, int j, int k) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), i * j * k, f);
    fclose(f);
}


template <int M, typename T=float>
void print_array_1d(T x[M]) {

    float x_cast[M]; 
    cast_1d<M, T, float>(x, x_cast);

    if (M < 6) {
        std::cout << "[ ";
        for (int i = 0; i < M; i++) {
            printf(" %010.6f ", x_cast[i]);
        }
        std::cout << " ]";
    } else {
        std::cout << "[ ";
        for (int i = 0; i < 3; i++) {
            printf(" %010.6f ", x_cast[i]);
        }
        std::cout << " ... ";
        for (int i = M - 3; i < M; i++) {
            printf(" %010.6f ", x_cast[i]);
        }
        std::cout << " ]";
    }

    std::cout << std::endl;
}

template <int M, int N, typename T=float>
void print_array_2d(T x[M][N]) {

    float x_cast[M][N];
    cast_2d<M, N, T, float>(x, x_cast);

    if ( (M < 6 && N < 6) || (M < 6 && N >= 6)) {
        std::cout << "[ ";
        for (int i = 0; i < M; i++) {
            if(i > 0) {
                std::cout << "  ";
            }
            print_array_1d<float, N>(x_cast[i]);
            if (i < M - 1) {
                std::cout << std::endl;
            }
        }
        std::cout << " ]" << std::endl;
    }
    else{
        std::cout << "[ ";
        for (int i = 0; i < 3; i++) {
            if(i > 0) {
                std::cout << "  ";
            }
            print_array_1d<float, N>(x_cast[i]);
            if (i < M - 1) {
                std::cout << std::endl;
            }
        }
        std::cout << "  ." << std::endl;
        std::cout << "  ." << std::endl;
        std::cout << "  ." << std::endl;
        for (int i = M - 3; i < M; i++) {
            if(i > 0) {
                std::cout << "  ";
            }
            print_array_1d<float, N>(x_cast[i]);
            if (i < M - 1) {
                std::cout << std::endl;
            }
        }
        std::cout << " ]" << std::endl;
    }

    std::cout << std::endl;
}

// Activation //

template <typename T>
T activation_elu(T x) {

    const T alpha = T(1.0);

    if (x > 0) {
        return x;
    } else {
        return alpha * (m_exp(x) - T(1.0));
    }
}

template <typename T>
T activation_hardtanh(T x) {

    const T min_val = T(-1.0);
    const T max_val = T(1.0);

    if (x < min_val) {
        return min_val;
    } else if (x > max_val) {
        return max_val;
    } else {
        return x;
    }
}

template <typename T>
T activation_leakyrelu(T x) {

    const T negative_slope = T(0.1);

    if (x >= 0) {
        return x;
    } else {
        return x * negative_slope;
    }
}

template <typename T>
T activation_relu(T x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

template <typename T>
T activation_gelu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    const T sqrt_2 = m_sqrt(T(2));
    const T sqrt_2_recip = T(1) / sqrt_2;
    const T one_half = T(0.5);
    T out = x * one_half * (T(1) + m_erf(x * sqrt_2_recip));
    return out;
}

template <typename T>
T activation_gelu_approx_1(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    const T sqrt_2_div_pi = m_sqrt(T(2) / m_pi());
    const T c = T(0.044715);
    const T one_half = T(0.5);
    T out = one_half * x * (1 + m_tanh(sqrt_2_div_pi * (x + c * m_pow(x, T(3)))));
    return out;
}

template <typename T>
T activation_gelu_approx_2(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
    const T c = T(1.702);
    T sigmod_arg = c * x;
    T sigmod_out = T(1.0) / (T(1.0) + m_exp(-sigmod_arg));
    T out = x * sigmod_arg;
    return out;
}

template <typename T>
T activation_sigmoid(T x) {
    return T(1.0) / (T(1.0) + m_exp(-x));
}

template <typename T>
T activation_silu(T x) {
    return x * (T(1.0) / (T(1.0) + m_exp(-x)));
}

template <typename T>
T activation_tanh(T x) {
#if __FLOATING_POINT_MODEL__
    T out = m_tanh(x);
    return out;
#else
    T out = m_tanh(x);
    T out_fixed = (hls::signbit(x) != hls::signbit(out)) ? T(-out) : out;
    return out_fixed;
#endif
}

template <typename T>
T activation_softsign(T x) {
    return x / (T(1.0) + m_abs(x));
}

template <typename T>
T activation_sin(T x) {
    return m_sin(x);
}

template <typename T>
T activation_cos(T x) {
    return m_cos(x);
}

template <typename T>
T activation_identity(T x) {
    return x;
}

// enum for activation function
enum ACTIVATION_FUNCTION {
    ELU,
    HARDTANH,
    LEAKYRELU,
    RELU,
    GELU,
    GELU_APPROX_1,
    GELU_APPROX_2,
    SIGMOID,
    SILU,
    TANH,
    SOFTSIGN,
    SIN,
    COS,
    IDENTITY
};

template <
    typename T,
    const int size,
    T (*activation_func)(T),
    const int block_size = 1>
class Activation {

    static_assert(size > 0, "size must be greater than 0");
    static_assert(block_size > 0, "block_size must be greater than 0");
    static_assert(size % block_size == 0, "size must be divisible by block_size");

public:
    void forward(T x_in[size], T x_out[size]) {
#pragma HLS INLINE off
#pragma HLS array_partition variable = x_in cyclic factor = block_size dim = 1
#pragma HLS array_partition variable = x_out cyclic factor = block_size dim = 1
        for (int i = 0; i < size; i += block_size) {
            #pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
                x_out[i + j] = activation_func(x_in[i + j]);
            }
        }
    }
};

// Linear //

template <typename T,
          const int in_size,
          const int out_size,
          const int block_size_in = 1,
          const int block_size_out = 1>
class Linear {

    static_assert(in_size > 0, "in_size must be greater than 0");
    static_assert(out_size > 0, "out_size must be greater than 0");
    static_assert(block_size_in > 0, "block_size_in must be greater than 0");
    static_assert(block_size_out > 0, "block_size_out must be greater than 0");
    static_assert(in_size % block_size_in == 0, "in_size must be divisible by block_size_in");
    static_assert(out_size % block_size_out == 0, "out_size must be divisible by block_size_out");

public:
    T weight[out_size][in_size] = {0};
    T bias[out_size] = {0};



    void load_params(T weight_in[out_size][in_size],
                     T bias_in[out_size]) {
#pragma HLS array_partition variable = weight cyclic factor = block_size_out dim = 1
#pragma HLS array_partition variable = weight cyclic factor = block_size_in dim = 2
#pragma HLS array_partition variable = bias cyclic factor = block_size_out dim = 1
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < in_size; j++) {
                weight[i][j] = weight_in[i][j];
            }
            bias[i] = bias_in[i];
        }
    }

    void forward(T input[in_size], T output[out_size]) {
#pragma HLS INLINE off

#pragma HLS array_partition variable = input cyclic factor = block_size_in dim = 1
#pragma HLS array_partition variable = output cyclic factor = block_size_out dim = 1

        T temp_sum[block_size_out];
#pragma HLS ARRAY_PARTITION variable = temp_sum complete

        // BLOCK_OUT
        for (int i = 0; i < out_size; i += block_size_out) {
            // BLOCK_IN
            for (int j = 0; j < in_size; j += block_size_in) {
#pragma HLS PIPELINE
                // TEMP_SUM_ZERO_LOOP
                for (int k = 0; k < block_size_out; k++) {
#pragma HLS UNROLL
                    temp_sum[k] = 0;
                }
                // SUM_OUTER
                for (int k = 0; k < block_size_out; k++) {
#pragma HLS UNROLL
                    // SUM_INNER
                    for (int l = 0; l < block_size_in; l++) {
#pragma HLS UNROLL
                        temp_sum[k] += weight[i + k][j + l] * input[j + l];
                    }
                }
                // WRITE_LOOP
                for (int k = 0; k < block_size_out; k++) {
#pragma HLS UNROLL
                    // check if first block itteration
                    // if first block itteration, write bias
                    if (j == 0) {
                        output[i + k] = bias[i + k];
                    }
                    output[i + k] += temp_sum[k];
                }
            }
        }
    }
};

// Normalization //

template <typename T,
          const int size,
          const int block_size = 1>
class LayerNorm {
    static_assert(size > 0, "size must be greater than 0");
    static_assert(block_size > 0, "block_size must be greater than 0");
    static_assert(size % block_size == 0, "size must be divisible by block_size");

public:
    T scale[size] = {T(0.0)};
    T bias[size] = {T(0.0)};

    void load_params(T scale_in[size], T bias_in[size]) {
#pragma HLS array_partition variable = scale cyclic factor = block_size dim = 1
#pragma HLS array_partition variable = bias cyclic factor = block_size dim = 1
        for (int i = 0; i < size; i++) {
            scale[i] = scale_in[i];
            bias[i] = bias_in[i];
        }
    }
    void forward(T x_in[size], T x_out[size]) {
#pragma HLS INLINE off
#pragma HLS array_partition variable = x_in cyclic factor = block_size dim = 1
#pragma HLS array_partition variable = x_out cyclic factor = block_size dim = 1

        T mean = T(0.0);
        T variance = T(0.0);
    }
};

// Architecture //

template <
    typename T,
    const int size,
    const int block_size = 1>
class SirenSine {

    static_assert(size > 0, "size must be greater than 0");
    static_assert(block_size > 0, "block_size must be greater than 0");
    static_assert(size % block_size == 0, "size must be divisible by block_size");

public:
    T w0 = T(1.0);

    void load_params(T w0_in) {
        w0 = w0_in;
    };

    void forward(T x_in[size], T x_out[size]) {
#pragma HLS inline off
#pragma HLS array_partition variable = x_in cyclic factor = block_size dim = 1
#pragma HLS array_partition variable = x_out cyclic factor = block_size dim = 1
        for (int i = 0; i < size; i += block_size) {
#pragma HLS PIPELINE II = 1
            for (int j = 0; j < block_size; j++) {
                x_out[i + j] = activation_sin(x_in[i + j] * w0);
            }
        }
    };
};

template <typename T,
          const int in_size,
          const int out_size,
          const int block_size_in = 1,
          const int block_size_out = 1>
class SirenLayer {
public:
    // #pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor = block_size_out dim = 1
    // #pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor = block_size_in dim = 2
    // #pragma HLS ARRAY_PARTITION variable = bias type = cyclic factor = block_size_out dim = 1

    Linear<T, in_size, out_size, block_size_in, block_size_out> linear;
    SirenSine<T, out_size, block_size_out> sine;

    void load_params(T weight_in[out_size][in_size], T bias_in[out_size], T w0_in) {
        linear.load_params(weight_in, bias_in);
        sine.load_params(w0_in);
    };

    void forward(T x_in[in_size], T x_out[out_size]) {
        linear.forward(x_in, x_out);
        sine.forward(x_out, x_out);
    };
};

template <
    typename T,
    const int model_in_dim,
    const int model_out_dim,
    const int model_hidden_dim,
    const int num_hidden_layers,
    T (*activation_func)(T) = activation_identity,
    int p_in = 1,
    int p_hidden = 1,
    int p_out = 1>
class SirenNet {

    // parameter checks //
    static_assert(model_in_dim > 0, "model_in_dim must be greater than 0");
    static_assert(model_out_dim > 0, "model_out_dim must be greater than 0");
    static_assert(model_hidden_dim > 0, "model_hidden_dim must be greater than 0");
    static_assert(num_hidden_layers > 0, "num_hidden_layers must be greater than 0");

    static_assert(p_in > 0, "p_in must be greater than 0");
    static_assert(p_hidden > 0, "p_hidden must be greater than 0");
    static_assert(p_out > 0, "p_out must be greater than 0");

    static_assert(model_in_dim % p_in == 0, "model_in_dim must be divisible by p_in");
    static_assert(model_hidden_dim % p_hidden == 0, "model_hidden_dim must be divisible by p_hidden");
    static_assert(model_out_dim % p_out == 0, "model_out_dim must be divisible by p_out");

public:
    SirenLayer<T, model_hidden_dim, model_hidden_dim, p_hidden, p_hidden> hidden_layers[num_hidden_layers];
    SirenLayer<T, model_in_dim, model_hidden_dim, p_in, p_hidden> input_layer;
    Linear<T, model_hidden_dim, model_out_dim, p_hidden, p_out> output_layer;
    Activation<T, model_out_dim, activation_func, p_out> output_activation;

    void load_params(
        T input_layer_weight[model_hidden_dim][model_in_dim],
        T input_layer_bias[model_hidden_dim],
        T input_layer_w0,
        T hidden_layers_weight[num_hidden_layers][model_hidden_dim][model_hidden_dim],
        T hidden_layers_bias[num_hidden_layers][model_hidden_dim],
        T hidden_layers_w0[num_hidden_layers],
        T output_layer_weight[model_out_dim][model_hidden_dim],
        T output_layer_bias[model_out_dim]) {
        input_layer.load_params(input_layer_weight, input_layer_bias, input_layer_w0);
        for (int i = 0; i < num_hidden_layers; i++) {
            hidden_layers[i].load_params(hidden_layers_weight[i], hidden_layers_bias[i], hidden_layers_w0[i]);
        }
        output_layer.load_params(output_layer_weight, output_layer_bias);
    };

    void forward(T x_in[model_in_dim],
                 T x_out[model_out_dim]) {
        T input_layer_buffer[model_hidden_dim];
        T hidden_layer_buffers[num_hidden_layers][model_hidden_dim];
        T output_layer_buffer[model_out_dim];

        input_layer.forward(x_in, input_layer_buffer);
        for (int i = 0; i < num_hidden_layers; i++) {
            if (i == 0) {
                hidden_layers[i].forward(input_layer_buffer, hidden_layer_buffers[i]);
            } else {
                hidden_layers[i].forward(hidden_layer_buffers[i - 1], hidden_layer_buffers[i]);
            }
        }
        output_layer.forward(hidden_layer_buffers[num_hidden_layers - 1], output_layer_buffer);
        output_activation.forward(output_layer_buffer, x_out);
    };
};
