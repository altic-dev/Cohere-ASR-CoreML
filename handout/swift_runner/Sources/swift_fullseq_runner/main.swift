import CoreML
import Foundation

struct FullSeqInput: Decodable {
    let encoder_hidden_states_shape: [Int]
    let encoder_hidden_states_flat: [Float]
    let prompt_ids: [Int]
    let eos_token_id: Int?
    let pad_token_id: Int?
    let id_to_token: [String]
    let max_len: Int
}

func makeFloatArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
    for i in 0..<values.count { arr[i] = NSNumber(value: values[i]) }
    return arr
}

func makeIntArray(shape: [Int], values: [Int32]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .int32)
    for i in 0..<values.count { arr[i] = NSNumber(value: values[i]) }
    return arr
}

func argmaxSlice(_ arr: MLMultiArray, tokenIndex: Int, maxLen: Int, vocabSize: Int) -> Int32 {
    // Layout assumed contiguous [1, maxLen, vocabSize]
    let offset = tokenIndex * vocabSize
    var best = arr[offset].doubleValue
    var bestIdx = 0
    for i in 1..<vocabSize {
        let v = arr[offset + i].doubleValue
        if v > best {
            best = v
            bestIdx = i
        }
    }
    return Int32(bestIdx)
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
    return pieces.joined().replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
}

@main
struct Main {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            print("Usage: swift_fullseq_runner <model.mlpackage> <swift_fullseq_input.json> [max_new_tokens] [trace_json_path]")
            return
        }
        let modelURL = URL(fileURLWithPath: args[1])
        let inputURL = URL(fileURLWithPath: args[2])
        let maxNewTokens = args.count >= 4 ? Int(args[3]) ?? 64 : 64
        let tracePath = args.count >= 5 ? args[4] : ""

        let payload = try JSONDecoder().decode(FullSeqInput.self, from: Data(contentsOf: inputURL))
        let encoderHidden = try makeFloatArray(
            shape: payload.encoder_hidden_states_shape,
            values: payload.encoder_hidden_states_flat
        )

        var inputIds = Array(repeating: Int32(payload.pad_token_id ?? 0), count: payload.max_len)
        var attnMask = Array(repeating: Int32(0), count: payload.max_len)
        for (i, tid) in payload.prompt_ids.enumerated() where i < payload.max_len {
            inputIds[i] = Int32(tid)
            attnMask[i] = 1
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let compiled = try MLModel.compileModel(at: modelURL)
        let model = try MLModel(contentsOf: compiled, configuration: config)

        let vocabSize = payload.id_to_token.count
        var curIdx = payload.prompt_ids.count - 1
        var stepTopTokens: [Int32] = []
        let t0 = Date()
        for _ in 0..<maxNewTokens {
            if curIdx + 1 >= payload.max_len { break }
            let idsArr = try makeIntArray(shape: [1, payload.max_len], values: inputIds)
            let maskArr = try makeIntArray(shape: [1, payload.max_len], values: attnMask)
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                "input_ids": MLFeatureValue(multiArray: idsArr),
                "decoder_attention_mask": MLFeatureValue(multiArray: maskArr),
            ])
            let out = try model.prediction(from: provider)
            guard let outName = out.featureNames.first,
                  let logits = out.featureValue(for: outName)?.multiArrayValue
            else {
                throw NSError(domain: "swift_fullseq_runner", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing logits"])
            }
            let next = argmaxSlice(logits, tokenIndex: curIdx, maxLen: payload.max_len, vocabSize: vocabSize)
            stepTopTokens.append(next)
            curIdx += 1
            inputIds[curIdx] = next
            attnMask[curIdx] = 1
            if let eos = payload.eos_token_id, Int(next) == eos { break }
        }
        let inferMs = Date().timeIntervalSince(t0) * 1000
        let used = Array(inputIds[0...curIdx])
        let text = decodePieces(ids: used, vocab: payload.id_to_token, eos: payload.eos_token_id, pad: payload.pad_token_id)
        print("infer_ms=\(String(format: "%.2f", inferMs))")
        print("generated_token_count=\(curIdx + 1)")
        print("prompt_token_count=\(payload.prompt_ids.count)")
        print("decoded_text=\(text)")
        if !tracePath.isEmpty {
            let trace: [String: Any] = [
                "generated_ids": used.map(Int.init),
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
