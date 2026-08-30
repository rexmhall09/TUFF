import Foundation
import Darwin
import Metal

/// One `mmap`'d window of `model_weights.bin`, wrapped in an `MTLBuffer`.
final class ResidentBuffer {
    let buffer: MTLBuffer
    let fileOffset: UInt64
    let residentSize: UInt64
    let mappedLength: Int

    /// `mmap` the page-aligned window covering `[fileOffset, fileOffset + residentSize)`
    /// inside the file at `fileURL`. The wrapped `MTLBuffer` starts at the
    /// (sub-page) offset within the mapping so the resident bytes start at
    /// byte 0 of the buffer.
    init(fileURL: URL,
                fileOffset: UInt64,
                residentSize: UInt64,
                device: MTLDevice,
                fileDescriptor: Int32? = nil) throws {
        let pageSize = Int(getpagesize())

        let fd = fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(fileURL.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        var info = stat()
        guard fstat(fd, &info) == 0 else {
            throw ModelError.posixFailed(call: "fstat(\(fileURL.path))", errno: errno)
        }
        guard (info.st_mode & S_IFMT) == S_IFREG, info.st_size >= 0 else {
            throw ModelError.indexCorrupt(detail: "resident weights are not a regular file")
        }
        let (fileEnd, endOverflow) = fileOffset.addingReportingOverflow(residentSize)
        guard residentSize > 0, !endOverflow, fileEnd <= UInt64(info.st_size),
              fileOffset <= UInt64(Int64.max) else {
            throw ModelError.indexCorrupt(detail: "resident mapping exceeds the weights file")
        }

        let alignedOffset = (fileOffset / UInt64(pageSize)) * UInt64(pageSize)
        let sliceShift = Int(fileOffset - alignedOffset)
        guard residentSize <= UInt64(Int.max - sliceShift) else {
            throw ModelError.indexCorrupt(detail: "resident mapping exceeds addressable memory")
        }
        let mappedLen = sliceShift + Int(residentSize)
        let mapped = mmap(nil, mappedLen, PROT_READ, MAP_PRIVATE,
                          fd, off_t(alignedOffset))
        if mapped == MAP_FAILED {
            throw ModelError.posixFailed(call: "mmap", errno: errno)
        }
        let base = mapped!

        _ = posix_madvise(base, mappedLen, POSIX_MADV_RANDOM)

        let sliceStart = base.advanced(by: sliceShift)

        // Capture pointer + length for the deallocator. Do NOT capture self
        // here — that would create a retain cycle through the MTLBuffer.
        nonisolated(unsafe) let captureBase = base
        let captureLen = mappedLen
        guard let buf = device.makeBuffer(
            bytesNoCopy: sliceStart,
            length: Int(residentSize),
            options: .storageModeShared,
            deallocator: { _, _ in
                munmap(captureBase, captureLen)
            }
        ) else {
            munmap(base, mappedLen)
            throw ModelError.residentBufferWrapFailed
        }

        self.buffer = buf
        self.fileOffset = fileOffset
        self.residentSize = residentSize
        self.mappedLength = mappedLen
    }
}

/// Tensor-safe collection of resident mappings. Metal limits the length of a
/// single buffer even when it wraps existing virtual memory, so a dense model
/// can legitimately be larger than `MTLDevice.maxBufferLength`. Regions are
/// cut only between complete index entries: a tensor's weights, scales, and
/// biases always remain in the same buffer.
final class ResidentBufferSet {
    struct RegionPlan: Equatable, Sendable {
        let fileOffset: UInt64
        let size: UInt64
        let tensorNames: [String]

        var endOffset: UInt64 { fileOffset + size }
    }

    struct ResolvedEntry {
        let buffer: MTLBuffer
        let regionFileOffset: UInt64
    }

    private let buffers: [ResidentBuffer]
    private let regionByTensorName: [String: Int]

    init(fileURL: URL,
         index: ResidentIndex,
         device: MTLDevice,
         fileDescriptor: Int32? = nil) throws {
        let plans = try Self.planRegions(
            entries: Array(index.entries.values),
            payloadOffset: index.header.indexSize,
            payloadSize: index.header.residentSize,
            maximumLength: UInt64(device.maxBufferLength))

        var opened: [ResidentBuffer] = []
        opened.reserveCapacity(plans.count)
        var lookup: [String: Int] = [:]
        lookup.reserveCapacity(index.entries.count)
        for (regionIndex, plan) in plans.enumerated() {
            opened.append(try ResidentBuffer(
                fileURL: fileURL,
                fileOffset: plan.fileOffset,
                residentSize: plan.size,
                device: device,
                fileDescriptor: fileDescriptor))
            for name in plan.tensorNames {
                lookup[name] = regionIndex
            }
        }
        self.buffers = opened
        self.regionByTensorName = lookup
    }

    func resolve(tensorName: String) throws -> ResolvedEntry {
        guard let regionIndex = regionByTensorName[tensorName] else {
            throw ModelError.tensorNotFound(name: tensorName)
        }
        let region = buffers[regionIndex]
        return ResolvedEntry(buffer: region.buffer,
                             regionFileOffset: region.fileOffset)
    }

    /// Pure planning step kept separate so boundary and oversized-tensor
    /// behavior can be tested without allocating large Metal buffers.
    static func planRegions(entries: [ResidentIndexEntry],
                            payloadOffset: UInt64,
                            payloadSize: UInt64,
                            maximumLength: UInt64) throws -> [RegionPlan] {
        guard maximumLength > 0 else {
            throw ModelError.residentBufferWrapFailed
        }
        let (payloadEnd, payloadOverflow) = payloadOffset.addingReportingOverflow(payloadSize)
        guard !payloadOverflow else {
            throw ModelError.indexCorrupt(detail: "resident payload range overflows UInt64")
        }

        struct SpannedEntry {
            let entry: ResidentIndexEntry
            let start: UInt64
            let end: UInt64
        }
        func fieldEnd(_ offset: UInt64, _ size: UInt64,
                      name: String, field: String) throws -> UInt64? {
            if size == 0 {
                guard offset == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "\(name).\(field) has an absent nonzero offset")
                }
                return nil
            }
            let (value, overflow) = offset.addingReportingOverflow(size)
            guard !overflow, offset >= payloadOffset, value <= payloadEnd else {
                throw ModelError.indexCorrupt(
                    detail: "\(name).\(field) exceeds the resident payload")
            }
            return value
        }

        let spanned = try entries.map { entry -> SpannedEntry in
            var starts: [UInt64] = []
            var ends: [UInt64] = []
            if let value = try fieldEnd(entry.fileOffset, entry.sizeBytes,
                                        name: entry.name, field: "weights") {
                starts.append(entry.fileOffset); ends.append(value)
            }
            if let value = try fieldEnd(entry.scaleOffset, entry.scaleSize,
                                        name: entry.name, field: "scales") {
                starts.append(entry.scaleOffset); ends.append(value)
            }
            if let value = try fieldEnd(entry.biasOffset, entry.biasSize,
                                        name: entry.name, field: "biases") {
                starts.append(entry.biasOffset); ends.append(value)
            }
            guard let start = starts.min(), let finish = ends.max() else {
                throw ModelError.indexCorrupt(detail: "\(entry.name) has no resident bytes")
            }
            guard finish - start <= maximumLength else {
                throw ModelError.indexCorrupt(
                    detail: "\(entry.name) spans \(finish - start) bytes, exceeding Metal's maximum buffer length \(maximumLength)")
            }
            return SpannedEntry(entry: entry, start: start, end: finish)
        }.sorted {
            if $0.start != $1.start { return $0.start < $1.start }
            if $0.end != $1.end { return $0.end < $1.end }
            return $0.entry.name < $1.entry.name
        }

        var plans: [RegionPlan] = []
        var currentStart: UInt64?
        var currentEnd: UInt64 = 0
        var currentNames: [String] = []
        func appendCurrent() {
            guard let start = currentStart else { return }
            plans.append(RegionPlan(fileOffset: start,
                                    size: currentEnd - start,
                                    tensorNames: currentNames))
        }
        for item in spanned {
            if let start = currentStart {
                let proposedEnd = max(currentEnd, item.end)
                if proposedEnd - start <= maximumLength {
                    currentEnd = proposedEnd
                    currentNames.append(item.entry.name)
                    continue
                }
                appendCurrent()
            }
            currentStart = item.start
            currentEnd = item.end
            currentNames = [item.entry.name]
        }
        appendCurrent()
        return plans
    }
}
