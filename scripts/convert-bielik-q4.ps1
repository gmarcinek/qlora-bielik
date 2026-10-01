$sourceModel = Resolve-Path "models/Bielik-11B-v3.0-Instruct"
$artifacts = Resolve-Path "artifacts"
$f16Model = Join-Path $artifacts "bielik-f16.gguf"
$q4Model = Join-Path $artifacts "bielik-q4_k_m.gguf"
$image = "ghcr.io/ggml-org/llama.cpp:full"

if (Test-Path $q4Model) {
    throw "Target already exists: $q4Model"
}

docker run --rm --entrypoint python3 `
    -v "${sourceModel}:/models/Bielik-11B-v3.0-Instruct:ro" `
    -v "${artifacts}:/artifacts" `
    $image /app/convert_hf_to_gguf.py /models/Bielik-11B-v3.0-Instruct `
    --outfile /artifacts/bielik-f16.gguf --outtype f16 --use-temp-file

docker run --rm --entrypoint /app/llama-quantize `
    -v "${artifacts}:/artifacts" `
    $image /artifacts/bielik-f16.gguf /artifacts/bielik-q4_k_m.gguf Q4_K_M