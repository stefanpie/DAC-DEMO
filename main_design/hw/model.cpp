#include "model.h"

// F_TYPE input_layer_weight[HIDDEN_DIM][INPUT_DIM];
// F_TYPE input_layer_bias[HIDDEN_DIM];
// F_TYPE input_layer_w0[1];
// F_TYPE hidden_layers_weight[NUM_HIDDEN_LAYERS][HIDDEN_DIM][HIDDEN_DIM];
// F_TYPE hidden_layers_bias[NUM_HIDDEN_LAYERS][HIDDEN_DIM];
// F_TYPE hidden_layers_w0[NUM_HIDDEN_LAYERS];
// F_TYPE output_layer_weight[OUTPUT_DIM][HIDDEN_DIM];
// F_TYPE output_layer_bias[OUTPUT_DIM];

// void load_parameters(
//     F_TYPE input_layer_weight_in[HIDDEN_DIM][INPUT_DIM],
//     F_TYPE input_layer_bias_in[HIDDEN_DIM],
//     F_TYPE input_layer_w0_in[1],
//     F_TYPE hidden_layers_weight_in[NUM_HIDDEN_LAYERS][HIDDEN_DIM][HIDDEN_DIM],
//     F_TYPE hidden_layers_bias_in[NUM_HIDDEN_LAYERS][HIDDEN_DIM],
//     F_TYPE hidden_layers_w0_in[NUM_HIDDEN_LAYERS],
//     F_TYPE output_layer_weight_in[OUTPUT_DIM][HIDDEN_DIM],
//     F_TYPE output_layer_bias_in[OUTPUT_DIM]
// ) {
//     copy_2d<HIDDEN_DIM, INPUT_DIM>(input_layer_weight_in, input_layer_weight);
//     copy_1d<HIDDEN_DIM>(input_layer_bias_in, input_layer_bias);
//     copy_1d<1>(input_layer_w0_in, input_layer_w0);
//     copy_3d<NUM_HIDDEN_LAYERS, HIDDEN_DIM, HIDDEN_DIM>(hidden_layers_weight_in, hidden_layers_weight);
//     copy_2d<NUM_HIDDEN_LAYERS, HIDDEN_DIM>(hidden_layers_bias_in, hidden_layers_bias);
//     copy_1d<NUM_HIDDEN_LAYERS>(hidden_layers_w0_in, hidden_layers_w0);
//     copy_2d<OUTPUT_DIM, HIDDEN_DIM>(output_layer_weight_in, output_layer_weight);
//     copy_1d<OUTPUT_DIM>(output_layer_bias_in, output_layer_bias);
// }

constexpr unsigned int AXI_XFER_BIT_WIDTH = 256;

SirenNet<F_TYPE, INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_HIDDEN_LAYERS, activation_identity, 1, 1, 1> siren_net;

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
    F_TYPE output_layer_bias_in[OUTPUT_DIM]) {

    #pragma HLS interface m_axi depth=1 port=x_in offset=slave bundle=inout1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=x_out offset=slave bundle=inout1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=load_parameters_flag offset=slave bundle=inout1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH

    #pragma HLS interface m_axi depth=1 port=input_layer_weight_in offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=input_layer_bias_in offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=input_layer_w0_in offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=hidden_layers_weight_in offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=hidden_layers_bias_in offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=hidden_layers_w0_in offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=output_layer_weight_in offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=output_layer_bias_in offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH



    if (load_parameters_flag) {
        siren_net.load_params(
            input_layer_weight_in,
            input_layer_bias_in,
            input_layer_w0_in[0],
            hidden_layers_weight_in,
            hidden_layers_bias_in,
            hidden_layers_w0_in,
            output_layer_weight_in,
            output_layer_bias_in
        );
    }
    
    siren_net.forward(x_in, x_out);
}