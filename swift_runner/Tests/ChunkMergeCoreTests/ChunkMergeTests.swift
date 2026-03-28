import XCTest
@testable import ChunkMergeCore

final class ChunkMergeTests: XCTestCase {
    func testWindowStartsSingleWhenShort() {
        let w = ChunkMerge.windowStarts(totalSamples: 100, windowSamples: 480_000, overlapSamples: 80_000)
        XCTAssertEqual(w, [0])
    }

    func testWindowStartsOverlapping() {
        let w = ChunkMerge.windowStarts(totalSamples: 100, windowSamples: 40, overlapSamples: 10)
        XCTAssertEqual(w, [0, 30, 60])
    }

    func testWindowStartsTailAligned() {
        let w = ChunkMerge.windowStarts(totalSamples: 95, windowSamples: 40, overlapSamples: 10)
        XCTAssertEqual(w, [0, 30, 55])
    }

    func testMergeDropsDuplicateWordPrefix() {
        let m = ChunkMerge.mergeTranscriptChunks(["hello world", "world how are you"])
        XCTAssertEqual(m, "hello world how are you")
    }

    func testMergeNoOverlapConcatenates() {
        let m = ChunkMerge.mergeTranscriptChunks(["alpha", "beta"])
        XCTAssertEqual(m, "alpha beta")
    }

    func testMergeBoundaryWord() {
        let m = ChunkMerge.mergeTranscriptChunks(["the quick brown", "quick brown fox"])
        XCTAssertEqual(m, "the quick brown fox")
    }
}
