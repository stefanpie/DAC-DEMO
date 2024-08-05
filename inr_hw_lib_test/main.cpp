#include "./main.h"

bool test_linear_layer() {

    const int in_size = 32;
    const int out_size = 16;

    // load input data
    float linear_in_float[in_size];
    F_TYPE linear_in_fixed[in_size];

    load_data_1d<in_size>("./test_data/linear_layer_in.bin", linear_in_float);
    cast_1d<in_size, float, F_TYPE>(linear_in_float, linear_in_fixed);

    // load parameters
    float linear_weight_float[out_size][in_size];
    float linear_bias_float[out_size];

    F_TYPE linear_weight_fixed[out_size][in_size];
    F_TYPE linear_bias_fixed[out_size];

    load_data_2d<out_size, in_size>("./test_data/linear_layer_weight.bin", linear_weight_float);
    load_data_1d<out_size>("./test_data/linear_layer_bias.bin", linear_bias_float);

    cast_2d<out_size, in_size, float, F_TYPE>(linear_weight_float, linear_weight_fixed);
    cast_1d<out_size, float, F_TYPE>(linear_bias_float, linear_bias_fixed);

    // load output data
    float linear_out_gold_float[out_size];
    load_data_1d<out_size>("./test_data/linear_layer_out.bin", linear_out_gold_float);

    // kernel data
    float linear_out_kernel_float[out_size];
    F_TYPE linear_out_kernel_fixed[out_size];

    // kernel execution
    Linear<F_TYPE, in_size, out_size> linear_kernel;
    linear_kernel.load_params(linear_weight_fixed, linear_bias_fixed);
    linear_kernel.forward(linear_in_fixed, linear_out_kernel_fixed);

    cast_1d<out_size, F_TYPE, float>(linear_out_kernel_fixed, linear_out_kernel_float);

    // compare output
    bool pass = true;
    float test_delta = 1e-3;
    for (int i = 0; i < out_size; i++) {
        // fabs
        float diff = fabs(linear_out_kernel_float[i] - linear_out_gold_float[i]);
        if (diff > test_delta) {
            pass = false;
        }
    }

    return pass;
}

bool test_activations() {

    const int in_size = 32;

    // load input data
    float activation_in_float[in_size];
    F_TYPE activation_in_fixed[in_size];

    load_data_1d<in_size>("./test_data/activation_in.bin", activation_in_float);
    cast_1d<in_size, float, F_TYPE>(activation_in_float, activation_in_fixed);

    // load output data
    float activation_out_elu_gold_float[in_size];
    float activation_out_hardtanh_gold_float[in_size];
    float activation_out_leakyrelu_gold_float[in_size];
    float activation_out_relu_gold_float[in_size];
    float activation_out_gelu_gold_float[in_size];
    float activation_out_sigmoid_gold_float[in_size];
    float activation_out_silu_gold_float[in_size];
    float activation_out_tanh_gold_float[in_size];
    float activation_out_softsign_gold_float[in_size];
    float activation_out_sin_gold_float[in_size];
    float activation_out_cos_gold_float[in_size];

    load_data_1d<in_size>("./test_data/activation_out_elu.bin", activation_out_elu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_hardtanh.bin", activation_out_hardtanh_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_leakyrelu.bin", activation_out_leakyrelu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_relu.bin", activation_out_relu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_gelu.bin", activation_out_gelu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_sigmoid.bin", activation_out_sigmoid_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_silu.bin", activation_out_silu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_tanh.bin", activation_out_tanh_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_softsign.bin", activation_out_softsign_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_sin.bin", activation_out_sin_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_cos.bin", activation_out_cos_gold_float);

    // kernel data
    float activation_out_elu_kernel_float[in_size];
    float activation_out_hardtanh_kernel_float[in_size];
    float activation_out_leakyrelu_kernel_float[in_size];
    float activation_out_relu_kernel_float[in_size];
    float activation_out_gelu_kernel_float[in_size];
    float activation_out_sigmoid_kernel_float[in_size];
    float activation_out_silu_kernel_float[in_size];
    float activation_out_tanh_kernel_float[in_size];
    float activation_out_softsign_kernel_float[in_size];
    float activation_out_sin_kernel_float[in_size];
    float activation_out_cos_kernel_float[in_size];
    F_TYPE activation_out_elu_kernel_fixed[in_size];
    F_TYPE activation_out_hardtanh_kernel_fixed[in_size];
    F_TYPE activation_out_leakyrelu_kernel_fixed[in_size];
    F_TYPE activation_out_relu_kernel_fixed[in_size];
    F_TYPE activation_out_gelu_kernel_fixed[in_size];
    F_TYPE activation_out_sigmoid_kernel_fixed[in_size];
    F_TYPE activation_out_silu_kernel_fixed[in_size];
    F_TYPE activation_out_tanh_kernel_fixed[in_size];
    F_TYPE activation_out_softsign_kernel_fixed[in_size];
    F_TYPE activation_out_sin_kernel_fixed[in_size];
    F_TYPE activation_out_cos_kernel_fixed[in_size];

    // kernel execution
    for (int i = 0; i < in_size; i++) {
        activation_out_elu_kernel_fixed[i] = activation_elu<F_TYPE>(activation_in_fixed[i]);
        activation_out_hardtanh_kernel_fixed[i] = activation_hardtanh<F_TYPE>(activation_in_fixed[i]);
        activation_out_leakyrelu_kernel_fixed[i] = activation_leakyrelu<F_TYPE>(activation_in_fixed[i]);
        activation_out_relu_kernel_fixed[i] = activation_relu<F_TYPE>(activation_in_fixed[i]);
        activation_out_gelu_kernel_fixed[i] = activation_gelu<F_TYPE>(activation_in_fixed[i]);
        activation_out_sigmoid_kernel_fixed[i] = activation_sigmoid<F_TYPE>(activation_in_fixed[i]);
        activation_out_silu_kernel_fixed[i] = activation_silu<F_TYPE>(activation_in_fixed[i]);
        activation_out_tanh_kernel_fixed[i] = activation_tanh<F_TYPE>(activation_in_fixed[i]);
        activation_out_softsign_kernel_fixed[i] = activation_softsign<F_TYPE>(activation_in_fixed[i]);
        activation_out_sin_kernel_fixed[i] = activation_sin<F_TYPE>(activation_in_fixed[i]);
        activation_out_cos_kernel_fixed[i] = activation_cos<F_TYPE>(activation_in_fixed[i]);
    }

    cast_1d<in_size, F_TYPE, float>(activation_out_elu_kernel_fixed, activation_out_elu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_hardtanh_kernel_fixed, activation_out_hardtanh_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_leakyrelu_kernel_fixed, activation_out_leakyrelu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_relu_kernel_fixed, activation_out_relu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_gelu_kernel_fixed, activation_out_gelu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_sigmoid_kernel_fixed, activation_out_sigmoid_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_silu_kernel_fixed, activation_out_silu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_tanh_kernel_fixed, activation_out_tanh_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_softsign_kernel_fixed, activation_out_softsign_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_sin_kernel_fixed, activation_out_sin_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_cos_kernel_fixed, activation_out_cos_kernel_float);

    // compare output
    bool pass = true;
    float test_delta = 1e-3;
    for (int i = 0; i < in_size; i++) {
        float diff_elu = fabs(activation_out_elu_gold_float[i] - activation_out_elu_kernel_float[i]);
        float diff_hardtanh = fabs(activation_out_hardtanh_gold_float[i] - activation_out_hardtanh_kernel_float[i]);
        float diff_leakyrelu = fabs(activation_out_leakyrelu_gold_float[i] - activation_out_leakyrelu_kernel_float[i]);
        float diff_relu = fabs(activation_out_relu_gold_float[i] - activation_out_relu_kernel_float[i]);
        float diff_gelu = fabs(activation_out_gelu_gold_float[i] - activation_out_gelu_kernel_float[i]);
        float diff_sigmoid = fabs(activation_out_sigmoid_gold_float[i] - activation_out_sigmoid_kernel_float[i]);
        float diff_silu = fabs(activation_out_silu_gold_float[i] - activation_out_silu_kernel_float[i]);
        float diff_tanh = fabs(activation_out_tanh_gold_float[i] - activation_out_tanh_kernel_float[i]);
        float diff_softsign = fabs(activation_out_softsign_gold_float[i] - activation_out_softsign_kernel_float[i]);
        float diff_sin = fabs(activation_out_sin_gold_float[i] - activation_out_sin_kernel_float[i]);
        float diff_cos = fabs(activation_out_cos_gold_float[i] - activation_out_cos_kernel_float[i]);

        if (diff_elu > test_delta) {
            pass = false;
        }
        if (diff_hardtanh > test_delta) {
            pass = false;
        }
        if (diff_leakyrelu > test_delta) {
            pass = false;
        }
        if (diff_relu > test_delta) {
            pass = false;
        }
        if (diff_gelu > test_delta) {
            pass = false;
        }
        if (diff_sigmoid > test_delta) {
            pass = false;
        }
        if (diff_silu > test_delta) {
            pass = false;
        }
        if (diff_tanh > test_delta) {
            pass = false;
        }
        if (diff_softsign > test_delta) {
            pass = false;
        }
        if (diff_sin > test_delta) {
            pass = false;
        }
        if (diff_cos > test_delta) {
            pass = false;
        }
    }

    return pass;
}

bool test_siren_net() {

    // architecture parameters

    const int dim_in = 2;
    const int dim_hidden = 256;
    const int dim_out = 3;
    const int num_hidden_layers = 4;

    // model parameters

    float input_layer_weight[dim_hidden][dim_in];
    float input_layer_bias[dim_hidden];
    float input_layer_w0[1];
    float hidden_layers_weight[num_hidden_layers][dim_hidden][dim_hidden];
    float hidden_layers_bias[num_hidden_layers][dim_hidden];
    float hidden_layers_w0[num_hidden_layers];
    float output_layer_weight[dim_out][dim_hidden];
    float output_layer_bias[dim_out];

    load_data_2d<dim_hidden, dim_in>("test_data/siren_input_layer_weight.bin", input_layer_weight);
    load_data_1d<dim_hidden>("test_data/siren_input_layer_bias.bin", input_layer_bias);
    load_data_1d<1>("test_data/siren_input_layer_w0.bin", input_layer_w0);
    load_data_3d<num_hidden_layers, dim_hidden, dim_hidden>("test_data/siren_hidden_layers_weight.bin", hidden_layers_weight);
    load_data_2d<num_hidden_layers, dim_hidden>("test_data/siren_hidden_layers_bias.bin", hidden_layers_bias);
    load_data_1d<num_hidden_layers>("test_data/siren_hidden_layers_w0.bin", hidden_layers_w0);
    load_data_2d<dim_out, dim_hidden>("test_data/siren_output_layer_weight.bin", output_layer_weight);
    load_data_1d<dim_out>("test_data/siren_output_layer_bias.bin", output_layer_bias);

    F_TYPE input_layer_weight_fixed[dim_hidden][dim_in];
    F_TYPE input_layer_bias_fixed[dim_hidden];
    F_TYPE input_layer_w0_fixed[1];
    F_TYPE hidden_layers_weight_fixed[num_hidden_layers][dim_hidden][dim_hidden];
    F_TYPE hidden_layers_bias_fixed[num_hidden_layers][dim_hidden];
    F_TYPE hidden_layers_w0_fixed[num_hidden_layers];
    F_TYPE output_layer_weight_fixed[dim_out][dim_hidden];
    F_TYPE output_layer_bias_fixed[dim_out];

    cast_2d<dim_hidden, dim_in, float, F_TYPE>(input_layer_weight, input_layer_weight_fixed);
    cast_1d<dim_hidden, float, F_TYPE>(input_layer_bias, input_layer_bias_fixed);
    cast_1d<1, float, F_TYPE>(input_layer_w0, input_layer_w0_fixed);
    cast_3d<num_hidden_layers, dim_hidden, dim_hidden, float, F_TYPE>(hidden_layers_weight, hidden_layers_weight_fixed);
    cast_2d<num_hidden_layers, dim_hidden, float, F_TYPE>(hidden_layers_bias, hidden_layers_bias_fixed);
    cast_1d<num_hidden_layers, float, F_TYPE>(hidden_layers_w0, hidden_layers_w0_fixed);
    cast_2d<dim_out, dim_hidden, float, F_TYPE>(output_layer_weight, output_layer_weight_fixed);
    cast_1d<dim_out, float, F_TYPE>(output_layer_bias, output_layer_bias_fixed);

    // model input
    float x_in[dim_in];
    F_TYPE x_in_fixed[dim_in];
    load_data_1d<dim_in>("test_data/siren_x_in.bin", x_in);
    cast_1d<dim_in, float, F_TYPE>(x_in, x_in_fixed);

    // golden output
    float x_out_golden[dim_out];
    load_data_1d<dim_out>("test_data/siren_x_out.bin", x_out_golden);

    // kernel
    float x_out_kernel_float[dim_out];
    F_TYPE x_out_kernel_fixed[dim_out];
    SirenNet<F_TYPE, dim_in, dim_out, dim_hidden, num_hidden_layers, activation_sigmoid, 1, 1, 1> siren_net;
    siren_net.load_params(
        input_layer_weight_fixed, input_layer_bias_fixed, input_layer_w0_fixed[0],
        hidden_layers_weight_fixed, hidden_layers_bias_fixed, hidden_layers_w0_fixed,
        output_layer_weight_fixed, output_layer_bias_fixed);

    siren_net.forward(x_in_fixed, x_out_kernel_fixed);

    cast_1d<dim_out, F_TYPE, float>(x_out_kernel_fixed, x_out_kernel_float);

    // compare
    bool pass = true;
    float test_delta = 1e-3;
    for (int i = 0; i < dim_out; i++) {
        float diff = fabs(x_out_kernel_float[i] - x_out_golden[i]);
        if (diff > test_delta) {
            pass = false;
        }
    }

    return pass;
}

int main() {
    printf("#######################\n");
    printf("### inr_hw_lib_test ###\n");
    printf("#######################\n");

    bool results_test_linear_layer = test_linear_layer();
    if (results_test_linear_layer) {
        printf("test_linear_layer: PASS\n");
    } else {
        printf("test_linear_layer: FAIL\n");
    }

    bool results_test_activations = test_activations();
    if (results_test_activations) {
        printf("test_activations: PASS\n");
    } else {
        printf("test_activations: FAIL\n");
    }

    bool results_test_siren_net = test_siren_net();
    if (results_test_siren_net) {
        printf("test_siren_net: PASS\n");
    } else {
        printf("test_siren_net: FAIL\n");
    }

}