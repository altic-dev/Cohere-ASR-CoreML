import Foundation

/// Overlapping fixed windows for long-form ASR (matches typical time-based chunk + overlap policies).
public enum ChunkMerge {
    /// Start indices for windows of length `windowSamples` with `overlapSamples` overlap between consecutive windows.
    public static func windowStarts(totalSamples: Int, windowSamples: Int, overlapSamples: Int) -> [Int] {
        guard totalSamples > 0, windowSamples > 0 else { return [] }
        guard overlapSamples >= 0, overlapSamples < windowSamples else { return [] }
        if totalSamples <= windowSamples { return [0] }
        let stride = windowSamples - overlapSamples
        var starts: [Int] = []
        var s = 0
        while s + windowSamples < totalSamples {
            starts.append(s)
            s += stride
        }
        let lastStart = totalSamples - windowSamples
        if starts.isEmpty || starts.last! < lastStart {
            starts.append(lastStart)
        }
        return starts
    }

    /// Stitch segment transcripts by dropping duplicated word n-grams at chunk boundaries.
    public static func mergeTranscriptChunks(_ parts: [String]) -> String {
        guard let first = parts.first else { return "" }
        var acc = first
        for next in parts.dropFirst() {
            acc = mergeTwo(acc, next)
        }
        return acc.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func mergeTwo(_ a: String, _ b: String) -> String {
        let wa = a.split(separator: " ").map(String.init)
        let wb = b.split(separator: " ").map(String.init)
        if wa.isEmpty { return b }
        if wb.isEmpty { return a }
        let maxJ = min(wa.count, wb.count, 48)
        for j in stride(from: maxJ, through: 1, by: -1) {
            if Array(wa.suffix(j)) == Array(wb.prefix(j)) {
                let rest = wb.dropFirst(j).joined(separator: " ")
                if rest.isEmpty { return a }
                return a + " " + rest
            }
        }
        return a + " " + b
    }
}
