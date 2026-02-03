import onnxruntime
import time
import os
import numpy as np

def load_onnx_models(encoder_path, ctc_path):
    """步骤 1: 加载 ONNX 音频编码器和 CTC Head"""
    # print("\n[1] 加载 ONNX Models (Encoder + CTC)...")
    
    t_start = time.perf_counter()
    session_opts = onnxruntime.SessionOptions()
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    encoder_sess = onnxruntime.InferenceSession(
        encoder_path, 
        sess_options=session_opts, 
        providers=['DmlExecutionProvider', 'CPUExecutionProvider']
    )
    
    ctc_sess = onnxruntime.InferenceSession(
        ctc_path, 
        sess_options=session_opts, 
        providers=['DmlExecutionProvider', 'CPUExecutionProvider']
    )
    
    t_cost = time.perf_counter() - t_start
    
    return encoder_sess, ctc_sess, t_cost

def encode_audio(audio, encoder_sess):
    """使用 ONNX Encoder 获取 LLM 嵌入和 CTC 特征"""
    
    # Check expected input type
    # 'tensor(float16)' -> float16
    input_type = encoder_sess.get_inputs()[0].type
    if 'float16' in input_type:
        dtype = np.float16
        print(f"   [Debug] Model expects FP16 input. (Type: {input_type})")
    else:
        dtype = np.float32
        print(f"   [Debug] Model expects FP32 input. (Type: {input_type})")

    # [FIX] 如果输入使用了 int16 范围 (如来自 nano_audio)，需要归一化到 [-1, 1]
    # 判断依据：如果最大值 > 100，则认为是 int16 范围
    if np.max(np.abs(audio)) > 100:
        print("   [Auto-Fix] Converting int16 range to [-1, 1]...")
        audio = audio / 32768.0

    # Reshape: (1, 1, audio_len) and cast
    audio_input = audio.astype(dtype).reshape(1, 1, -1)
    
    in_names = [x.name for x in encoder_sess.get_inputs()]
    out_names = [x.name for x in encoder_sess.get_outputs()]
    
    # 输入: audio
    # 输出: enc_output, adaptor_output
    input_feed = {
        in_names[0]: onnxruntime.OrtValue.ortvalue_from_numpy(audio_input, 'cpu', 0)
    }
    
    outputs = encoder_sess.run_with_ort_values(out_names, input_feed)
    
    # Output 0: enc_output [1, T_enc, 512] (For CTC)
    enc_output = outputs[0].numpy()
    
    # Output 1: adaptor_output [1, T_llm, 1024] (For LLM)
    # MUST cast to float32 because LLM bindings (ctypes/llama.cpp) usually expect float* (32-bit)
    audio_embd = outputs[1].numpy().squeeze(0).astype(np.float32)
    
    return audio_embd, enc_output
