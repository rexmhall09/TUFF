import Foundation

@MainActor
public enum MetricFormat {
    private static let memoryFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter
    }()

    private static let storageFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useMB, .useGB, .useTB]
        formatter.includesUnit = true
        formatter.isAdaptive = true
        return formatter
    }()

    public static func seconds(_ value: Double?) -> String {
        guard let value else { return "\u{2014}" }
        if value < 1 { return String(format: "%.0f ms", value * 1000) }
        return String(format: "%.2f s", value)
    }

    public static func milliseconds(_ value: Double?) -> String {
        guard let value else { return "\u{2014}" }
        return "\(value.formatted(.number.precision(.fractionLength(1)))) ms"
    }

    public static func rate(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    public static func percent(_ value: Double) -> String {
        String(format: "%.1f%%", value)
    }

    public static func perToken(_ value: Double) -> String {
        "\(value.formatted(.number.precision(.fractionLength(1))))/tok"
    }

    public static func megabytesPerToken(_ value: Double) -> String {
        "\(value.formatted(.number.precision(.fractionLength(1)))) MB/tok"
    }

    public static func memory(_ bytes: UInt64?) -> String {
        guard let bytes else { return "\u{2014}" }
        return memoryFormatter.string(fromByteCount: Int64(bytes))
    }

    public static func storage(_ bytes: UInt64) -> String {
        storageFormatter.string(fromByteCount: Int64(clamping: bytes))
    }


    /// Token counts as "12K" once they are large enough that the exact figure
    /// is noise. Used by the context meter, which is an estimate anyway.
    public static func tokens(_ count: Int) -> String {
        if count < 1_000 { return "\(count)" }
        let thousands = Double(count) / 1_000
        return thousands < 10
            ? String(format: "%.1fK", thousands)
            : "\(Int(thousands.rounded()))K"
    }
}
