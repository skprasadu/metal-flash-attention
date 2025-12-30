//
//  LeadingDimensionsCacheKeyTest.swift
//  metal-flash-attention
//
//  Created by Krishna Srinivasamurthy on 12/30/25.
//
import XCTest
import FlashAttention

final class LeadingDimensionsCacheKeyTest: XCTestCase {
  func testPipelineCacheDifferentiatesLeadingDimensions() throws {
    let M: UInt32 = 128
    let N: UInt32 = 128
    let K: UInt32 = 128

    let transposeState = (A: false, B: false)
    let prec = (A: GEMMOperandPrecision.FP16,
                B: GEMMOperandPrecision.FP16,
                C: GEMMOperandPrecision.FP16)

    // Descriptor 1 (tight)
    var d1 = GEMMDescriptor()
    d1.loadPreviousC = false
    d1.matrixDimensions = (M: M, N: N, K: K)
    d1.memoryPrecisions = prec
    d1.transposeState = transposeState
    d1.leadingDimensions = (A: K, B: N, C: N)

    // ✅ Add this block right here:
    var d3 = d1
    d3.leadingDimensions = nil
    GEMMKernel.register(descriptor: d3)
    XCTAssertEqual(
        d1,
        d3,
        "Nil leadingDimensions should equal explicit expected leadingDimensions."
    )
    XCTAssertNotNil(GEMMKernel.pipelineCache[d3])
    
    // Descriptor 2 (padded stride)
    var d2 = d1
    d2.leadingDimensions = (A: K + 16, B: N + 32, C: N + 64)

    // Force compilation/caching of both
    GEMMKernel.register(descriptor: d1)
    GEMMKernel.register(descriptor: d2)

    // If the key is wrong, these may alias and you’ll silently reuse the wrong pipeline.
    XCTAssertNotEqual(d1, d2, "Descriptors should not be equal if leadingDimensions differ.")

    // Stronger: ensure cache contains two distinct entries.
    // (Only do this if pipelineCache is accessible; otherwise just run both and validate outputs.)
    XCTAssertNotNil(GEMMKernel.pipelineCache[d1])
    XCTAssertNotNil(GEMMKernel.pipelineCache[d2])
  }
}
