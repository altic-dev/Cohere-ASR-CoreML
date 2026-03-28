@preconcurrency import AVFoundation
import ChunkMergeCore
import CoreML
import Foundation

struct Manifest: Decodable {
    let sample_rate: Int
    let precision: String?
    let quantize: String?
    let preemph: Float?
    let max_audio_samples: Int
    let max_audio_seconds: Double?
    let overlap_seconds: Double?
    let overlap_samples: Int?
    let max_feature_frames: Int
    let max_encoder_frames: Int
    let decoder_max_len: Int
    let default_max_new_tokens: Int
    let prompt_ids: [Int]
    let eos_token_id: Int?
    let pad_token_id: Int?
    let id_to_token: [String]
    let frontend: StageMeta
    let encoder: StageMeta
    let decoder: StageMeta
    let decoder_cached: CachedDecoderMeta?

    var isFp16: Bool { precision == "float16" }
}

struct StageMeta: Decodable {
    let package: String
    let inputs: [String]
    let outputs: [String]
}

struct CachedDecoderMeta: Decodable {
    let package: String
    let inputs: [String]
    let outputs: [String]
    let logits_output: String
    let cache_k_output: String
    let cache_v_output: String
    let num_layers: Int
    let num_heads: Int
    let head_dim: Int
}

struct CliArgs {
    let audioPath: String
    let artifactsDir: String
    let compiledCacheDir: String
    let computeMode: String
    let decoderMode: String
    let traceJsonPath: String?
    let maxNewTokens: Int?
    /// If set, overrides manifest `overlap_seconds` (or default 5s) for chunk stride.
    let overlapSecondsOverride: Double?
}

func parseArgs(_ args: [String]) throws -> CliArgs {
    var audioPath = ""
    var tracePath: String?
    var maxNewTokens: Int?
    var overlapSecondsOverride: Double?

    let cwd = FileManager.default.currentDirectoryPath
    let defaultArtifacts = URL(fileURLWithPath: cwd).appendingPathComponent("../artifacts").path
    var artifactsDir = defaultArtifacts
    var compiledCacheDir = URL(fileURLWithPath: defaultArtifacts).appendingPathComponent(".compiled").path
    var computeMode = "ane"
    var decoderMode = "cached"

    var i = 1
    while i < args.count {
        let key = args[i]
        if key == "--audio", i + 1 < args.count {
            audioPath = args[i + 1]
            i += 2
            continue
        }
        if key == "--artifacts-dir", i + 1 < args.count {
            artifactsDir = args[i + 1]
            i += 2
            continue
        }
        if key == "--trace-json", i + 1 < args.count {
            tracePath = args[i + 1]
            i += 2
            continue
        }
        if key == "--compiled-cache-dir", i + 1 < args.count {
            compiledCacheDir = args[i + 1]
            i += 2
            continue
        }
        if key == "--compute", i + 1 < args.count {
            computeMode = args[i + 1].lowercased()
            i += 2
            continue
        }
        if key == "--decoder-mode", i + 1 < args.count {
            decoderMode = args[i + 1].lowercased()
            i += 2
            continue
        }
        if key == "--max-new-tokens", i + 1 < args.count {
            maxNewTokens = Int(args[i + 1])
            i += 2
            continue
        }
        if key == "--overlap-seconds", i + 1 < args.count {
            overlapSecondsOverride = Double(args[i + 1])
            i += 2
            continue
        }
        throw NSError(domain: "pure_coreml_asr_cli", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "Unknown or incomplete argument: \(key)"
        ])
    }

    if audioPath.isEmpty {
        throw NSError(domain: "pure_coreml_asr_cli", code: 3, userInfo: [
            NSLocalizedDescriptionKey: "Usage: swift run pure_coreml_asr_cli --audio <path.wav> [--trace-json <path>] [--max-new-tokens N] [--overlap-seconds SEC] [--artifacts-dir <path>] [--compiled-cache-dir <path>] [--compute cpu|gpu|ane|all]"
        ])
    }

    return CliArgs(
        audioPath: audioPath,
        artifactsDir: artifactsDir,
        compiledCacheDir: compiledCacheDir,
        computeMode: computeMode,
        decoderMode: decoderMode,
        traceJsonPath: tracePath,
        maxNewTokens: maxNewTokens,
        overlapSecondsOverride: overlapSecondsOverride
    )
}

func loadManifest(artifactsDir: String) throws -> Manifest {
    let manifestURL = URL(fileURLWithPath: artifactsDir).appendingPathComponent("coreml_manifest.json")
    return try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
}

/// Overlap stride for long audio: CLI override, else manifest `overlap_samples` / `overlap_seconds`, else 5s (PyTorch-style default).
func effectiveOverlapSamples(manifest: Manifest, cli: CliArgs) -> Int {
    let maxS = manifest.max_audio_samples
    if maxS <= 1 { return 0 }
    if let o = cli.overlapSecondsOverride {
        let v = Int(round(Double(manifest.sample_rate) * o))
        return min(max(0, v), maxS - 1)
    }
    if let os = manifest.overlap_samples, os >= 0 {
        return min(os, maxS - 1)
    }
    let sec = manifest.overlap_seconds ?? 5.0
    let v = Int(round(Double(manifest.sample_rate) * sec))
    return min(max(0, v), maxS - 1)
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

func toContiguous(_ arr: MLMultiArray) throws -> MLMultiArray {
    let shape = arr.shape.map { $0.intValue }
    let totalCount = shape.reduce(1, *)
    let strides = arr.strides.map { $0.intValue }
    let isContiguous: Bool = {
        var expected = 1
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[i] != expected { return false }
            expected *= shape[i]
        }
        return true
    }()
    if isContiguous { return arr }
    let out = try MLMultiArray(shape: arr.shape, dataType: arr.dataType)
    if arr.dataType == .float16 {
        let srcBase = arr.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount * 2)
        let dstBase = out.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
        if shape.count == 3 {
            var outIdx = 0
            for b in 0..<shape[0] {
                for t in 0..<shape[1] {
                    let base = b * strides[0] + t * strides[1]
                    if strides[2] == 1 {
                        memcpy(dstBase + outIdx, srcBase + base, shape[2] * MemoryLayout<UInt16>.size)
                        outIdx += shape[2]
                    } else {
                        for h in 0..<shape[2] {
                            dstBase[outIdx] = srcBase[base + h * strides[2]]
                            outIdx += 1
                        }
                    }
                }
            }
        }
    } else {
        let srcBase = arr.dataPointer.bindMemory(to: Float.self, capacity: totalCount * 2)
        let dstBase = out.dataPointer.bindMemory(to: Float.self, capacity: totalCount)
        if shape.count == 3 {
            var outIdx = 0
            for b in 0..<shape[0] {
                for t in 0..<shape[1] {
                    let base = b * strides[0] + t * strides[1]
                    if strides[2] == 1 {
                        memcpy(dstBase + outIdx, srcBase + base, shape[2] * MemoryLayout<Float>.size)
                        outIdx += shape[2]
                    } else {
                        for h in 0..<shape[2] {
                            dstBase[outIdx] = srcBase[base + h * strides[2]]
                            outIdx += 1
                        }
                    }
                }
            }
        }
    }
    return out
}

func makeFloat16Array(shape: [Int], values: [Float]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .float16)
    let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: values.count)
    for i in 0..<values.count {
        var f16 = Float16(values[i])
        withUnsafeBytes(of: &f16) { buf in
            ptr[i] = buf.load(as: UInt16.self)
        }
    }
    return arr
}

final class FastFeatureProvider: MLFeatureProvider {
    private var dict: [String: MLFeatureValue]
    let featureNames: Set<String>

    init(_ pairs: [(String, MLMultiArray)]) {
        var d = [String: MLFeatureValue]()
        d.reserveCapacity(pairs.count)
        var names = Set<String>()
        names.reserveCapacity(pairs.count)
        for (k, v) in pairs {
            d[k] = MLFeatureValue(multiArray: v)
            names.insert(k)
        }
        self.dict = d
        self.featureNames = names
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        dict[featureName]
    }

    func update(_ key: String, _ arr: MLMultiArray) {
        dict[key] = MLFeatureValue(multiArray: arr)
    }
}

func readAudioMono16k(path: String, targetSampleRate: Int) throws -> [Float] {
    let srcURL = URL(fileURLWithPath: path)
    let file = try AVAudioFile(forReading: srcURL)
    let srcFormat = file.processingFormat

    guard let srcBuffer = AVAudioPCMBuffer(
        pcmFormat: srcFormat,
        frameCapacity: AVAudioFrameCount(file.length)
    ) else {
        throw NSError(domain: "pure_coreml_asr_cli", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate source audio buffer"])
    }
    try file.read(into: srcBuffer)

    guard let dstFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(targetSampleRate),
        channels: 1,
        interleaved: false
    ) else {
        throw NSError(domain: "pure_coreml_asr_cli", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create destination audio format"])
    }

    guard let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else {
        throw NSError(domain: "pure_coreml_asr_cli", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio converter"])
    }

    let ratio = dstFormat.sampleRate / srcFormat.sampleRate
    let cap = AVAudioFrameCount((Double(srcBuffer.frameLength) * ratio).rounded(.up) + 8)
    guard let dstBuffer = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: cap) else {
        throw NSError(domain: "pure_coreml_asr_cli", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate converted audio buffer"])
    }

    var consumed = false
    var convertError: NSError?
    _ = converter.convert(to: dstBuffer, error: &convertError) { _, outStatus in
        if consumed {
            outStatus.pointee = .endOfStream
            return nil
        }
        consumed = true
        outStatus.pointee = .haveData
        return srcBuffer
    }
    if let convertError {
        throw convertError
    }

    guard let channel = dstBuffer.floatChannelData?[0] else {
        throw NSError(domain: "pure_coreml_asr_cli", code: 8, userInfo: [NSLocalizedDescriptionKey: "No float channel data in converted audio"])
    }
    let count = Int(dstBuffer.frameLength)
    return Array(UnsafeBufferPointer(start: channel, count: count))
}

func applyPreemphasis(_ audio: inout [Float], rawLength: Int, coeff: Float) {
    let count = min(rawLength, audio.count)
    if count > 1 {
        for i in stride(from: count - 1, through: 1, by: -1) {
            audio[i] = audio[i] - coeff * audio[i - 1]
        }
    }
    for i in count..<audio.count {
        audio[i] = 0.0
    }
}

func argmax1D(_ arr: MLMultiArray) -> Int32 {
    let count = arr.count
    if arr.dataType == .float16 {
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        var bestIdx = 0
        var bestBits = ptr[0]
        for i in 1..<count {
            let bits = ptr[i]
            if Float16(bitPattern: bits) > Float16(bitPattern: bestBits) {
                bestBits = bits; bestIdx = i
            }
        }
        return Int32(bestIdx)
    } else {
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
        var bestIdx = 0
        var best = ptr[0]
        for i in 1..<count {
            if ptr[i] > best { best = ptr[i]; bestIdx = i }
        }
        return Int32(bestIdx)
    }
}

func argmaxSlice(_ arr: MLMultiArray, tokenIndex: Int, vocabSize: Int) -> Int32 {
    // Respect tensor strides; output is [1, T, V] but not guaranteed contiguous.
    let strideT = arr.strides[1].intValue
    let strideV = arr.strides[2].intValue
    let base = tokenIndex * strideT
    var best = arr[base].doubleValue
    var bestIdx = 0
    for i in 1..<vocabSize {
        let v = arr[base + i * strideV].doubleValue
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

func mapComputeUnits(_ mode: String) throws -> MLComputeUnits {
    switch mode {
    case "cpu":
        return .cpuOnly
    case "gpu":
        return .cpuAndGPU
    case "ane":
        return .cpuAndNeuralEngine
    case "all":
        return .all
    default:
        throw NSError(domain: "pure_coreml_asr_cli", code: 13, userInfo: [
            NSLocalizedDescriptionKey: "Invalid --compute mode '\(mode)'. Use cpu|gpu|ane|all."
        ])
    }
}

func loadModel(_ url: URL, compiledCacheRoot: URL, computeMode: String) throws -> (model: MLModel, compiledNow: Bool) {
    let config = MLModelConfiguration()
    config.computeUnits = try mapComputeUnits(computeMode)

    let fm = FileManager.default
    let attrs = try fm.attributesOfItem(atPath: url.path)
    let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
    let key = "\(url.deletingPathExtension().lastPathComponent)_\(computeMode)_\(Int64(mtime * 1000)).mlmodelc"
    let cachedCompiled = compiledCacheRoot.appendingPathComponent(key)
    try fm.createDirectory(at: compiledCacheRoot, withIntermediateDirectories: true)
    var compiledNow = false

    if !fm.fileExists(atPath: cachedCompiled.path) {
        let lockURL = compiledCacheRoot.appendingPathComponent("\(key).lock")
        while true {
            do {
                try fm.createDirectory(at: lockURL, withIntermediateDirectories: false)
                break
            } catch {
                if fm.fileExists(atPath: lockURL.path) {
                    Thread.sleep(forTimeInterval: 0.1)
                    continue
                }
                throw error
            }
        }
        defer { try? fm.removeItem(at: lockURL) }

        if !fm.fileExists(atPath: cachedCompiled.path) {
            let compiledTmp = try MLModel.compileModel(at: url)
            let staged = compiledCacheRoot.appendingPathComponent("\(key).staging.\(UUID().uuidString)")
            try? fm.removeItem(at: staged)
            try fm.copyItem(at: compiledTmp, to: staged)
            if fm.fileExists(atPath: cachedCompiled.path) {
                try? fm.removeItem(at: staged)
            } else {
                try fm.moveItem(at: staged, to: cachedCompiled)
                compiledNow = true
            }
        }
    }
    return (try MLModel(contentsOf: cachedCompiled, configuration: config), compiledNow)
}

@main
struct Main {
    static func main() throws {
        let cli = try parseArgs(CommandLine.arguments)
        let manifest = try loadManifest(artifactsDir: cli.artifactsDir)

        let artifactsURL = URL(fileURLWithPath: cli.artifactsDir)
        let compiledCacheURL = URL(fileURLWithPath: cli.compiledCacheDir)
        let useCached = cli.decoderMode == "cached"
        let frontendURL = artifactsURL.appendingPathComponent(manifest.frontend.package)
        let encoderURL = artifactsURL.appendingPathComponent(manifest.encoder.package)
        let decoderURL: URL
        if useCached, let dc = manifest.decoder_cached {
            decoderURL = artifactsURL.appendingPathComponent(dc.package)
        } else {
            decoderURL = artifactsURL.appendingPathComponent(manifest.decoder.package)
        }

        let tLoad0 = Date()
        let frontendLoaded = try loadModel(frontendURL, compiledCacheRoot: compiledCacheURL, computeMode: cli.computeMode)
        let encoderLoaded = try loadModel(encoderURL, compiledCacheRoot: compiledCacheURL, computeMode: cli.computeMode)
        let decoderLoaded = try loadModel(decoderURL, compiledCacheRoot: compiledCacheURL, computeMode: cli.computeMode)
        let loadMs = Date().timeIntervalSince(tLoad0) * 1000
        let frontendModel = frontendLoaded.model
        let encoderModel = encoderLoaded.model
        let decoderModel = decoderLoaded.model
        let needsAnewarmup = cli.computeMode == "ane"

        let tAudio0 = Date()
        let fullAudio = try readAudioMono16k(path: cli.audioPath, targetSampleRate: manifest.sample_rate)
        let audioMs = Date().timeIntervalSince(tAudio0) * 1000
        guard !fullAudio.isEmpty else {
            throw NSError(domain: "pure_coreml_asr_cli", code: 15, userInfo: [
                NSLocalizedDescriptionKey: "Decoded audio is empty",
            ])
        }

        let preemphCoeff = manifest.preemph ?? 0.97
        let overlapSamples = effectiveOverlapSamples(manifest: manifest, cli: cli)
        let starts = ChunkMerge.windowStarts(
            totalSamples: fullAudio.count,
            windowSamples: manifest.max_audio_samples,
            overlapSamples: overlapSamples
        )

        func buildFrontInputs(paddedAudio: [Float], rawLength: Int) throws -> MLDictionaryFeatureProvider {
            let audioArr = try makeFloatArray(shape: [1, manifest.max_audio_samples], values: paddedAudio)
            let audioLenArr = try makeIntArray(shape: [1], values: [Int32(rawLength)])
            return try MLDictionaryFeatureProvider(dictionary: [
                manifest.frontend.inputs[0]: MLFeatureValue(multiArray: audioArr),
                manifest.frontend.inputs[1]: MLFeatureValue(multiArray: audioLenArr),
            ])
        }

        func buildChunkAtIndex(_ chunkIndex: Int) throws -> MLDictionaryFeatureProvider {
            let start = starts[chunkIndex]
            let end = min(start + manifest.max_audio_samples, fullAudio.count)
            var segment = Array(fullAudio[start..<end])
            let rawLength = segment.count
            if segment.count < manifest.max_audio_samples {
                segment += Array(repeating: 0.0, count: manifest.max_audio_samples - segment.count)
            }
            applyPreemphasis(&segment, rawLength: rawLength, coeff: preemphCoeff)
            return try buildFrontInputs(paddedAudio: segment, rawLength: rawLength)
        }

        func runFrontendEncoder(frontInputs: MLDictionaryFeatureProvider) throws -> (frontMs: Double, encMs: Double, encoderHiddenF32: MLMultiArray, encoderValid: Int) {
            let tFront0 = Date()
            let frontOut = try frontendModel.prediction(from: frontInputs)
            let frontMs = Date().timeIntervalSince(tFront0) * 1000
            guard
                let inputFeatures = frontOut.featureValue(for: manifest.frontend.outputs[0])?.multiArrayValue,
                let featureLength = frontOut.featureValue(for: manifest.frontend.outputs[1])?.multiArrayValue
            else {
                throw NSError(domain: "pure_coreml_asr_cli", code: 9, userInfo: [NSLocalizedDescriptionKey: "Frontend outputs missing"])
            }

            let encInputs = try MLDictionaryFeatureProvider(dictionary: [
                manifest.encoder.inputs[0]: MLFeatureValue(multiArray: inputFeatures),
                manifest.encoder.inputs[1]: MLFeatureValue(multiArray: featureLength),
            ])

            let tEnc0 = Date()
            let encOut = try encoderModel.prediction(from: encInputs)
            let encMs = Date().timeIntervalSince(tEnc0) * 1000
            guard
                let encoderHidden = encOut.featureValue(for: manifest.encoder.outputs[0])?.multiArrayValue,
                let encoderLengthOut = encOut.featureValue(for: manifest.encoder.outputs[1])?.multiArrayValue
            else {
                throw NSError(domain: "pure_coreml_asr_cli", code: 10, userInfo: [NSLocalizedDescriptionKey: "Encoder outputs missing"])
            }
            let encoderHiddenF32 = try toContiguous(encoderHidden)
            let encoderValidRaw = Int(encoderLengthOut[0].doubleValue.rounded())
            let encoderValid = max(1, min(manifest.max_encoder_frames, encoderValidRaw))
            return (frontMs, encMs, encoderHiddenF32, encoderValid)
        }

        func runFullSeqPipeline(frontInputs: MLDictionaryFeatureProvider) throws -> (frontMs: Double, encMs: Double, decMs: Double, used: [Int32], stepTopTokens: [Int32]) {
            let fe = try runFrontendEncoder(frontInputs: frontInputs)
            let encoderHiddenF32 = fe.encoderHiddenF32
            let encoderValid = fe.encoderValid

            let vocabSize = manifest.id_to_token.count
            let maxNewTokens = cli.maxNewTokens ?? manifest.default_max_new_tokens
            var inputIds = Array(repeating: Int32(manifest.pad_token_id ?? 0), count: manifest.decoder_max_len)
            var attnMask = Array(repeating: Int32(0), count: manifest.decoder_max_len)
            for (i, tid) in manifest.prompt_ids.enumerated() where i < manifest.decoder_max_len {
                inputIds[i] = Int32(tid)
                attnMask[i] = 1
            }

            var crossMaskValues = Array(repeating: Float(-1e9), count: manifest.max_encoder_frames)
            for i in 0..<encoderValid { crossMaskValues[i] = 0.0 }
            let crossMaskArr: MLMultiArray
            if manifest.isFp16 {
                crossMaskArr = try makeFloat16Array(shape: [1, 1, 1, manifest.max_encoder_frames], values: crossMaskValues)
            } else {
                crossMaskArr = try makeFloatArray(shape: [1, 1, 1, manifest.max_encoder_frames], values: crossMaskValues)
            }

            var curIdx = manifest.prompt_ids.count - 1
            var stepTopTokens: [Int32] = []
            let tDec0 = Date()
            for _ in 0..<maxNewTokens {
                if curIdx + 1 >= manifest.decoder_max_len { break }
                let idsArr = try makeIntArray(shape: [1, manifest.decoder_max_len], values: inputIds)
                let maskArr = try makeIntArray(shape: [1, manifest.decoder_max_len], values: attnMask)
                let decInputs = try MLDictionaryFeatureProvider(dictionary: [
                    manifest.decoder.inputs[0]: MLFeatureValue(multiArray: encoderHiddenF32),
                    manifest.decoder.inputs[1]: MLFeatureValue(multiArray: idsArr),
                    manifest.decoder.inputs[2]: MLFeatureValue(multiArray: maskArr),
                    manifest.decoder.inputs[3]: MLFeatureValue(multiArray: crossMaskArr),
                ])
                let decOut = try decoderModel.prediction(from: decInputs)
                guard let logits = decOut.featureValue(for: manifest.decoder.outputs[0])?.multiArrayValue else {
                    throw NSError(domain: "pure_coreml_asr_cli", code: 11, userInfo: [NSLocalizedDescriptionKey: "Decoder logits missing"])
                }
                let next = argmaxSlice(logits, tokenIndex: curIdx, vocabSize: vocabSize)
                stepTopTokens.append(next)
                curIdx += 1
                inputIds[curIdx] = next
                attnMask[curIdx] = 1
                if let eos = manifest.eos_token_id, Int(next) == eos { break }
            }
            let decMs = Date().timeIntervalSince(tDec0) * 1000
            let used = Array(inputIds[0...curIdx])
            return (fe.frontMs, fe.encMs, decMs, used, stepTopTokens)
        }

        func runCachedPipeline(frontInputs: MLDictionaryFeatureProvider) throws -> (frontMs: Double, encMs: Double, decMs: Double, used: [Int32], stepTopTokens: [Int32]) {
            guard let dc = manifest.decoder_cached else {
                throw NSError(domain: "pure_coreml_asr_cli", code: 14, userInfo: [NSLocalizedDescriptionKey: "decoder_cached not found in manifest"])
            }
            let fe = try runFrontendEncoder(frontInputs: frontInputs)
            let encoderHiddenF32 = fe.encoderHiddenF32
            let encoderValid = fe.encoderValid

            let maxNewTokens = cli.maxNewTokens ?? manifest.default_max_new_tokens
            let cacheSize = dc.num_layers * dc.num_heads * manifest.decoder_max_len * dc.head_dim
            let cacheShape = [dc.num_layers, dc.num_heads, manifest.decoder_max_len, dc.head_dim] as [NSNumber]
            let cacheType: MLMultiArrayDataType = manifest.isFp16 ? .float16 : .float32
            let cacheBytesPerElem = manifest.isFp16 ? 2 : 4

            let vocabSize = manifest.id_to_token.count
            let logitsShape = [1, vocabSize] as [NSNumber]

            // Double-buffered caches: A and B swap each step
            let cacheKA = try MLMultiArray(shape: cacheShape, dataType: cacheType)
            let cacheVA = try MLMultiArray(shape: cacheShape, dataType: cacheType)
            let cacheKB = try MLMultiArray(shape: cacheShape, dataType: cacheType)
            let cacheVB = try MLMultiArray(shape: cacheShape, dataType: cacheType)
            let logitsBuf = try MLMultiArray(shape: logitsShape, dataType: cacheType)
            memset(cacheKA.dataPointer, 0, cacheSize * cacheBytesPerElem)
            memset(cacheVA.dataPointer, 0, cacheSize * cacheBytesPerElem)

            var useA = true  // current input caches are A

            var crossMaskValues = Array(repeating: Float(-1e9), count: manifest.max_encoder_frames)
            for i in 0..<encoderValid { crossMaskValues[i] = 0.0 }
            let crossMaskArr: MLMultiArray
            if manifest.isFp16 {
                crossMaskArr = try makeFloat16Array(shape: [1, 1, 1, manifest.max_encoder_frames], values: crossMaskValues)
            } else {
                crossMaskArr = try makeFloatArray(shape: [1, 1, 1, manifest.max_encoder_frames], values: crossMaskValues)
            }

            let inputIdArr = try MLMultiArray(shape: [1, 1], dataType: .int32)
            let stepArr = try MLMultiArray(shape: [1], dataType: .int32)

            let featureProvider = FastFeatureProvider([
                ("encoder_hidden_states", encoderHiddenF32),
                ("input_id", inputIdArr),
                ("cache_k", cacheKA),
                ("cache_v", cacheVA),
                ("step", stepArr),
                ("cross_attention_mask", crossMaskArr),
            ])

            let optsA = MLPredictionOptions()
            optsA.outputBackings = [
                dc.logits_output: logitsBuf,
                dc.cache_k_output: cacheKB,
                dc.cache_v_output: cacheVB,
            ]
            let optsB = MLPredictionOptions()
            optsB.outputBackings = [
                dc.logits_output: logitsBuf,
                dc.cache_k_output: cacheKA,
                dc.cache_v_output: cacheVA,
            ]

            func runOneStep(tokenId: Int32, stepIdx: Int32) throws -> MLMultiArray {
                inputIdArr[0] = NSNumber(value: tokenId)
                stepArr[0] = NSNumber(value: stepIdx)
                let inK = useA ? cacheKA : cacheKB
                let inV = useA ? cacheVA : cacheVB
                featureProvider.update("cache_k", inK)
                featureProvider.update("cache_v", inV)
                let opts = useA ? optsA : optsB
                _ = try decoderModel.prediction(from: featureProvider, options: opts)
                useA = !useA
                return logitsBuf
            }

            var stepTopTokens: [Int32] = []
            var generated = manifest.prompt_ids.map { Int32($0) }

            let tDec0 = Date()
            var lastLogits: MLMultiArray?
            for (i, tid) in manifest.prompt_ids.enumerated() {
                lastLogits = try runOneStep(tokenId: Int32(tid), stepIdx: Int32(i))
            }

            var curIdx = manifest.prompt_ids.count
            if let logits = lastLogits {
                let next = argmax1D(logits)
                stepTopTokens.append(next)
                generated.append(next)
                if let eos = manifest.eos_token_id, Int(next) == eos {
                    let decMs = Date().timeIntervalSince(tDec0) * 1000
                    return (fe.frontMs, fe.encMs, decMs, generated, stepTopTokens)
                }
            }

            for _ in 1..<maxNewTokens {
                if curIdx >= manifest.decoder_max_len { break }
                let logits = try runOneStep(tokenId: generated.last!, stepIdx: Int32(curIdx))
                let next = argmax1D(logits)
                stepTopTokens.append(next)
                generated.append(next)
                curIdx += 1
                if let eos = manifest.eos_token_id, Int(next) == eos { break }
            }
            let decMs = Date().timeIntervalSince(tDec0) * 1000
            return (fe.frontMs, fe.encMs, decMs, generated, stepTopTokens)
        }

        let runPipeline = useCached ? runCachedPipeline : runFullSeqPipeline
        if needsAnewarmup, !starts.isEmpty {
            _ = try runPipeline(try buildChunkAtIndex(0))
        }

        var totalFrontMs = 0.0
        var totalEncMs = 0.0
        var totalDecMs = 0.0
        var chunkTexts: [String] = []
        var lastUsed: [Int32] = []
        var lastStepTop: [Int32] = []

        for chunkIdx in 0..<starts.count {
            let fi = try buildChunkAtIndex(chunkIdx)
            let result = try runPipeline(fi)
            totalFrontMs += result.frontMs
            totalEncMs += result.encMs
            totalDecMs += result.decMs
            lastUsed = result.used
            lastStepTop = result.stepTopTokens
            let chunkText = decodePieces(
                ids: result.used,
                vocab: manifest.id_to_token,
                eos: manifest.eos_token_id,
                pad: manifest.pad_token_id
            )
            chunkTexts.append(chunkText)
        }

        let text = ChunkMerge.mergeTranscriptChunks(chunkTexts)
        let used = lastUsed
        let stepTopTokens = lastStepTop

        print("load_ms=\(String(format: "%.2f", loadMs))")
        print("precision=\(manifest.precision ?? "float32")")
        print("quantize=\(manifest.quantize ?? "none")")
        print("compute_mode=\(cli.computeMode)")
        print("decoder_mode=\(cli.decoderMode)")
        print("audio_ms=\(String(format: "%.2f", audioMs))")
        print("chunk_count=\(starts.count)")
        print("overlap_samples=\(overlapSamples)")
        print("frontend_ms_total=\(String(format: "%.2f", totalFrontMs))")
        print("encoder_ms_total=\(String(format: "%.2f", totalEncMs))")
        print("decoder_ms_total=\(String(format: "%.2f", totalDecMs))")
        if starts.count == 1 {
            print("frontend_ms=\(String(format: "%.2f", totalFrontMs))")
            print("encoder_ms=\(String(format: "%.2f", totalEncMs))")
            print("decoder_ms=\(String(format: "%.2f", totalDecMs))")
        }
        print("generated_token_count=\(used.count)")
        print("prompt_token_count=\(manifest.prompt_ids.count)")
        print("decoded_text=\(text)")

        if let tracePath = cli.traceJsonPath {
            let trace: [String: Any] = [
                "audio_file": cli.audioPath,
                "chunk_count": starts.count,
                "overlap_samples": overlapSamples,
                "chunk_texts": chunkTexts,
                "generated_ids": used.map(Int.init),
                "step_top_tokens": stepTopTokens.map(Int.init),
                "decoded_text": text,
            ]
            let traceData = try JSONSerialization.data(withJSONObject: trace, options: [.prettyPrinted, .sortedKeys])
            try traceData.write(to: URL(fileURLWithPath: tracePath))
            print("trace_json=\(tracePath)")
        }
    }
}
