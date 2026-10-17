"""一次本地推理即退出；加载前设置所属进程的内存硬限。"""

import contextlib
import json
import os
from pathlib import Path
import sys


def memory_limit(process=None, limit=None):
    limit = limit or int(os.environ['MONOPLANER_RAG_WORKER_BYTES'])
    if os.name != 'nt':
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        return None
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                    ('flags', wintypes.DWORD), ('min_set', ctypes.c_size_t),
                    ('max_set', ctypes.c_size_t), ('active', wintypes.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                    ('scheduling', wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

    class Limits(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                    ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                    ('peak_job', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.CreateJobObjectW(None, None)
    value = Limits()
    value.basic.flags = 0x2000 | 0x100 | 0x200  # 退出杀所属进程；进程与作业内存上限。
    value.process_memory = value.job_memory = limit
    if (not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(value),
                                                        ctypes.sizeof(value))
            or not kernel.AssignProcessToJobObject(
                handle, wintypes.HANDLE(int(process._handle)) if process else kernel.GetCurrentProcess())):
        raise RuntimeError('Cannot enforce the local model memory limit.')
    return handle


def infer(request):
    spec, payload, role = request['spec'], request['payload'], request['role']
    root = Path(spec['root'])
    if role == 'laya':
        import laya
        import onnxruntime as ort
        from laya.common import build_sequence, encode_text, render_options

        if set(ort.get_available_providers()) - {'CPUExecutionProvider', 'AzureExecutionProvider'}:
            raise ValueError('Laya qualification requires a CPU-only ONNX Runtime installation.')
        model = laya.load(str(root), backend='onnx', onnx_path=str(root / spec['model']),
                          expected_sha256=spec['sha256'])
        state, questions = payload['state'], payload['questions']
        head_max = spec.get('head_max_tokens', 192)
        for definition in questions.values():
            internal = {'t': definition['type'], 'ins': definition['instructions'],
                        'crit': definition.get('criteria')}
            _, markers, stats, state_stats = build_sequence(
                model.tok, state, internal, spec['max_tokens'], head_max,
                return_stats=True, return_truncation_stats=True,
            )
            cap = stats.get('tokens_per_option')
            options_cut = cap is not None and any(
                len(encode_text(model.tok, item, add_special_tokens=False)['input_ids']) > cap
                for item in render_options(internal)
            )
            if state_stats['truncated'] or len(markers) != stats['options'] or options_cut:
                raise ValueError('Laya evidence or options exceed the input window; nothing was truncated.')
        result = model.predict(state, questions, max_len=spec['max_tokens'], head_max_len=head_max)
        if result.get('usage', {}).get('truncated'):
            raise ValueError('Laya reported truncated input; result is unusable.')
        return result
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root / spec['tokenizer']))
    tokenizer.no_truncation()
    tokenizer.enable_padding(pad_id=spec.get('pad_id', 0), pad_token=spec.get('pad_token', '[PAD]'))
    pairs = [(payload['query'], text) for text in payload['texts']] if role == 'reranker' else payload['texts']
    encodings = tokenizer.encode_batch(pairs)
    if any(len(item.ids) > spec['max_tokens'] for item in encodings):
        raise ValueError('Model input exceeds the verified token window; nothing was truncated.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(root / spec['model']), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    arrays = {
        'input_ids': np.array([item.ids for item in encodings], dtype=np.int64),
        'attention_mask': np.array([item.attention_mask for item in encodings], dtype=np.int64),
        'token_type_ids': np.array([item.type_ids for item in encodings], dtype=np.int64),
    }
    result = session.run(None, {field.name: arrays[field.name] for field in session.get_inputs()})[0]
    if role == 'embedding':
        if result.ndim == 3:
            mask = arrays['attention_mask'][..., None]
            result = (result * mask).sum(1) / np.maximum(mask.sum(1), 1)
        result = result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
        return result.tolist()
    if result.ndim != 2 or result.shape[1] != 1:
        raise ValueError('Reranker requires one relevance logit per query/document pair.')
    if spec.get('activation', 'identity') == 'identity':
        return result[:, 0].tolist()
    if spec['activation'] != 'sigmoid':
        raise ValueError('Unknown reranker activation; use the pinned model contract.')
    return (1 / (1 + np.exp(-np.clip(result[:, 0], -60, 60)))).tolist()


def main():
    job = None
    try:
        if os.environ.get('MONOPLANER_RAG_PARENT_LIMIT') != '1':
            job = memory_limit()
        if '--count' in sys.argv:
            from tokenizers import Tokenizer

            request = json.loads(sys.stdin.buffer.readline(1024 * 1024 + 1))
            spec = request['spec']
            tokenizer = Tokenizer.from_file(str(Path(spec['root']) / spec['tokenizer']))
            tokenizer.no_truncation()
            tokenizer.no_padding()
            for line in sys.stdin.buffer:
                if len(line) > 512 * 1024:
                    raise ValueError('Tokenizer counting input exceeds its bound.')
                text = json.loads(line)
                encoded = tokenizer.encode(spec.get('document_prefix', '') + text)
                print(json.dumps({'tokens': len(encoded.ids)}), flush=True)
            return 0
        request = json.loads(sys.stdin.buffer.read(1024 * 1024 + 1))
        with contextlib.redirect_stdout(sys.stderr):
            result = infer(request)
        print(json.dumps({'result': result}, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
