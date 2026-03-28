import CoreML
import Foundation

struct SwiftInput: Decodable {
    let encoder_hidden_states_shape: [Int]
    let encoder_hidden_states_flat: [Float]
    let input_ids_shape: [Int]
    let input_ids_flat: [Int32]
    let reference_top_token: Int
    let tokenizer: TokenizerMeta
}

struct TokenizerMeta: Decodable {
    let bos_token_id: Int?
    let eos_token_id: Int?
    let pad_token_id: Int?
    let vocab_size: Int
    let id_to_token: [String]
}

enum RunnerError: Error {
    case badArgs
    case shapeMismatch(String)
    case emptyOutput
    case unsupportedOutputType(String)
}

func makeFloatMultiArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
    let count = shape.reduce(1, *)
    guard count == values.count else {
        throw RunnerError.shapeMismatch("Float array count mismatch: expected \(count), got \(values.count)")
    }
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
    for i in 0..<count {
        arr[i] = NSNumber(value: values[i])
    }
    return arr
}

func makeIntMultiArray(shape: [Int], values: [Int32]) throws -> MLMultiArray {
    let count = shape.reduce(1, *)
    guard count == values.count else {
        throw RunnerError.shapeMismatch("Int array count mismatch: expected \(count), got \(values.count)")
    }
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .int32)
    for i in 0..<count {
        arr[i] = NSNumber(value: values[i])
    }
    return arr
}

func argmax(_ arr: MLMultiArray) -> Int {
    let count = arr.count
    var bestIdx = 0
    var bestVal = arr[0].doubleValue
    for i in 1..<count {
        let v = arr[i].doubleValue
        if v > bestVal {
            bestVal = v
            bestIdx = i
        }
    }
    return bestIdx
}

func decodeTokenPieces(ids: [Int], tokenizer: TokenizerMeta) -> String {
    var pieces: [String] = []
    pieces.reserveCapacity(ids.count)
    for id in ids {
        if id < 0 || id >= tokenizer.id_to_token.count { continue }
        if let eos = tokenizer.eos_token_id, id == eos { break }
        if let bos = tokenizer.bos_token_id, id == bos { continue }
        if let pad = tokenizer.pad_token_id, id == pad { continue }
        let t = tokenizer.id_to_token[id]
        if t.hasPrefix("<") && t.hasSuffix(">") { continue }
        pieces.append(t)
    }
    let merged = pieces.joined()
    return merged.replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
}

@main
struct Main {
    static func main() throws {
        do {
            let args = CommandLine.arguments
            guard args.count == 3 else {
                print("Usage: swift_runner <path/to/model.mlpackage> <path/to/swift_input.json>")
                throw RunnerError.badArgs
            }

            let modelURL = URL(fileURLWithPath: args[1])
            let inputURL = URL(fileURLWithPath: args[2])

            let inputData = try Data(contentsOf: inputURL)
            let payload = try JSONDecoder().decode(SwiftInput.self, from: inputData)

            let encoderHiddenStates = try makeFloatMultiArray(
                shape: payload.encoder_hidden_states_shape, values: payload.encoder_hidden_states_flat
            )

            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine

            let tCompileStart = Date()
            let compiled = try MLModel.compileModel(at: modelURL)
            let compileMs = Date().timeIntervalSince(tCompileStart) * 1000.0

            let model = try MLModel(contentsOf: compiled, configuration: config)
            let inputConstraint = model.modelDescription.inputDescriptionsByName["input_ids"]?.multiArrayConstraint
            let shape = inputConstraint?.shape.map { $0.intValue } ?? []
            let fixedMaxLen = shape.count >= 2 ? shape[1] : payload.input_ids_flat.count

            let maxNewTokens = 96
            var generatedIds: [Int32] = payload.input_ids_flat
            let inferStart = Date()
            var firstOutName = ""
            var firstStepTopToken: Int?
            var stoppedForShapeLimit = false
            for _ in 0..<maxNewTokens {
                if generatedIds.count > fixedMaxLen {
                    stoppedForShapeLimit = true
                    break
                }
                let curInput = try makeIntMultiArray(shape: [1, generatedIds.count], values: generatedIds)
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "encoder_hidden_states": MLFeatureValue(multiArray: encoderHiddenStates),
                    "input_ids": MLFeatureValue(multiArray: curInput),
                ])
                let output = try model.prediction(from: provider)
                guard let outName = output.featureNames.first else {
                    throw RunnerError.emptyOutput
                }
                if firstOutName.isEmpty { firstOutName = outName }
                guard let outArray = output.featureValue(for: outName)?.multiArrayValue else {
                    throw RunnerError.unsupportedOutputType(outName)
                }
                let topToken = argmax(outArray)
                if firstStepTopToken == nil { firstStepTopToken = topToken }
                generatedIds.append(Int32(topToken))
                if let eos = payload.tokenizer.eos_token_id, topToken == eos {
                    break
                }
            }
            let inferMs = Date().timeIntervalSince(inferStart) * 1000.0
            let generatedInts = generatedIds.map(Int.init)
            let decodedText = decodeTokenPieces(ids: generatedInts, tokenizer: payload.tokenizer)

            let firstToken = firstStepTopToken ?? -1
            print("compile_ms=\(String(format: "%.2f", compileMs))")
            print("infer_ms=\(String(format: "%.2f", inferMs))")
            print("output_name=\(firstOutName)")
            print("swift_top_token=\(firstToken)")
            print("reference_top_token=\(payload.reference_top_token)")
            print("top_token_match=\(firstToken == payload.reference_top_token)")
            print("generated_token_count=\(generatedIds.count)")
            print("fixed_input_token_limit=\(fixedMaxLen)")
            print("stopped_for_shape_limit=\(stoppedForShapeLimit)")
            print("decoded_text=\(decodedText)")
        } catch {
            print("swift_runner_error=\(error)")
            throw error
        }
    }
}
