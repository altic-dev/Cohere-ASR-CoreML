import CoreML
import Foundation

struct CachedInput: Decodable {
    let encoder_hidden_states_shape: [Int]
    let encoder_hidden_states_flat: [Float]
    let prompt_ids: [Int]
    let bos_token_id: Int
    let eos_token_id: Int?
    let pad_token_id: Int?
    let vocab_size: Int
    let id_to_token: [String]
    let num_layers: Int
    let num_heads: Int
    let head_dim: Int
    let max_len: Int
    let logits_output_name: String
    let cache_k_output_name: String
    let cache_v_output_name: String
}

func makeFloatMultiArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
    let count = shape.reduce(1, *)
    guard count == values.count else {
        throw NSError(domain: "swift_cached_runner", code: 1, userInfo: [NSLocalizedDescriptionKey: "shape mismatch"])
    }
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
    for i in 0..<count { arr[i] = NSNumber(value: values[i]) }
    return arr
}

func zerosFloatArray(shape: [Int]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
    for i in 0..<arr.count { arr[i] = 0 }
    return arr
}

func makeIntArray(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .int32)
    for i in 0..<values.count { arr[i] = NSNumber(value: values[i]) }
    return arr
}

func argmax(_ arr: MLMultiArray) -> Int32 {
    var best = arr[0].doubleValue
    var idx = 0
    for i in 1..<arr.count {
        let v = arr[i].doubleValue
        if v > best {
            best = v
            idx = i
        }
    }
    return Int32(idx)
}

func decodePieces(ids: [Int32], vocab: [String], eos: Int?, pad: Int?) -> String {
    var pieces: [String] = []
    for t in ids {
        let id = Int(t)
        if let e = eos, id == e { break }
        if let p = pad, id == p { continue }
        if id < 0 || id >= vocab.count { continue }
        let piece = vocab[id]
        if piece.hasPrefix("<") && piece.hasSuffix(">") { continue }
        pieces.append(piece)
    }
    let merged = pieces.joined()
    return merged.replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
}

@main
struct Main {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            print("Usage: swift_cached_runner <model.mlpackage> <swift_cached_input.json> [max_new_tokens] [trace_json_path]")
            return
        }
        let modelURL = URL(fileURLWithPath: args[1])
        let inputURL = URL(fileURLWithPath: args[2])
        let maxNewTokens = args.count >= 4 ? Int(args[3]) ?? 96 : 96
        let tracePath = args.count >= 5 ? args[4] : ""

        let payload = try JSONDecoder().decode(CachedInput.self, from: Data(contentsOf: inputURL))
        let encoderHidden = try makeFloatMultiArray(
            shape: payload.encoder_hidden_states_shape,
            values: payload.encoder_hidden_states_flat
        )
        let cacheShape = [payload.num_layers, payload.num_heads, payload.max_len, payload.head_dim]
        var cacheK = try zerosFloatArray(shape: cacheShape)
        var cacheV = try zerosFloatArray(shape: cacheShape)

        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let compiled = try MLModel.compileModel(at: modelURL)
        let model = try MLModel(contentsOf: compiled, configuration: config)

        var generated: [Int32] = payload.prompt_ids.map { Int32($0) }
        var stepTopTokens: [Int32] = []
        let t0 = Date()
        var step = 0
        while step < maxNewTokens {
            if generated.count >= payload.max_len { break }
            let inputId = try makeIntArray([generated.last ?? 0], shape: [1, 1])
            let stepArr = try makeIntArray([Int32(generated.count - 1)], shape: [1])
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                "input_id": MLFeatureValue(multiArray: inputId),
                "cache_k": MLFeatureValue(multiArray: cacheK),
                "cache_v": MLFeatureValue(multiArray: cacheV),
                "step": MLFeatureValue(multiArray: stepArr),
            ])
            let out = try model.prediction(from: provider)
            guard let logitsArr = out.featureValue(for: payload.logits_output_name)?.multiArrayValue else {
                throw NSError(domain: "swift_cached_runner", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing logits output"])
            }
            guard let nextCacheK = out.featureValue(for: payload.cache_k_output_name)?.multiArrayValue,
                  let nextCacheV = out.featureValue(for: payload.cache_v_output_name)?.multiArrayValue else {
                throw NSError(domain: "swift_cached_runner", code: 3, userInfo: [NSLocalizedDescriptionKey: "missing cache outputs"])
            }
            cacheK = nextCacheK
            cacheV = nextCacheV
            let next = argmax(logitsArr)
            stepTopTokens.append(next)
            generated.append(next)
            if let eos = payload.eos_token_id, Int(next) == eos { break }
            step += 1
        }
        let inferMs = Date().timeIntervalSince(t0) * 1000
        let text = decodePieces(ids: generated, vocab: payload.id_to_token, eos: payload.eos_token_id, pad: payload.pad_token_id)
        print("infer_ms=\(String(format: "%.2f", inferMs))")
        print("generated_token_count=\(generated.count)")
        print("prompt_token_count=\(payload.prompt_ids.count)")
        print("decoded_text=\(text)")
        if !tracePath.isEmpty {
            let trace: [String: Any] = [
                "generated_ids": generated.map(Int.init),
                "step_top_tokens": stepTopTokens.map(Int.init),
                "prompt_ids": payload.prompt_ids,
                "decoded_text": text,
                "max_new_tokens": maxNewTokens,
            ]
            let data = try JSONSerialization.data(withJSONObject: trace, options: [.prettyPrinted])
            try data.write(to: URL(fileURLWithPath: tracePath))
            print("trace_json=\(tracePath)")
        }
    }
}
