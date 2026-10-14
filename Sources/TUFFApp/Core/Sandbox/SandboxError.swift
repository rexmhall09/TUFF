#if os(macOS)
import Foundation

/// Why the web sandbox could not start, or is not safe to use. The messages
/// are written for the person using the app; they carry a short tail of the
/// failing command, cleaned of control characters.
public enum SandboxError: Error, Equatable, Sendable, CustomStringConvertible {
    case unsupportedSystem
    case containerNotInstalled
    /// The tool is there, but the person or their group could replace it, or a
    /// folder above it, so it is not run.
    case containerNotTrusted
    case containerTooOld(found: String, minimum: String)
    case containerNotSigned(String)
    case containerCommandFailed(action: String, detail: String)
    case timedOut(action: String)
    case imageNotPublished
    case digestMismatch(expected: String, found: String?)
    case anotherInstanceRunning
    case unsafeMountPath
    case noFreePort
    case secretUnavailable(String)
    case tooManyAddresses
    case hostAddressesUnknown
    case notHealthy(String)
    case protectionNotVerified(String)
    case stoppedWhileStarting
    case notReady
    case stillStarting
    /// A files sandbox was asked to start with no folder to mount.
    case noFoldersToMount
    /// macOS refused to let the container tool read a folder.
    case folderAccessDenied
    /// A search folder's path cannot be given to `container run --mount`.
    case unmountableFolder

    public var description: String {
        switch self {
        case .unsupportedSystem:
            "The sandbox needs macOS 26 or newer."
        case .containerNotInstalled:
            "Apple container is not installed. Install it from github.com/apple/container/releases."
        case .containerNotTrusted:
            "TUFF will not run the container tool: it, or a folder above it, can be changed by someone "
                + "other than an administrator. Make it and every folder above it owned by root and "
                + "not writable by others (Terminal: ls -ld on each), or install Apple container again."
        case .containerTooOld(let found, let minimum):
            "Apple container \(found) is too old; the sandbox needs \(minimum) or newer."
        case .containerNotSigned(let detail):
            "The container tool at /usr/local/bin/container does not have a valid signature"
                + (detail.isEmpty ? "." : " (\(detail)).")
        case .containerCommandFailed(let action, let detail):
            detail.isEmpty ? "Could not \(action)." : "Could not \(action): \(detail)"
        case .timedOut(let action):
            "Timed out while trying to \(action)."
        case .imageNotPublished:
            "The sandbox image has not been published yet, so there is nothing to download."
        case .digestMismatch(let expected, let found):
            "The sandbox image is not the one this version of TUFF expects (wanted \(expected), found "
                + "\(found ?? "nothing")). It was not used."
        case .anotherInstanceRunning:
            "Another copy of TUFF is using the sandbox. Quit it first."
        case .unsafeMountPath:
            "TUFF's data folder has a comma in its path, which the container tool cannot mount."
        case .noFreePort:
            "No free local port was found for the sandbox."
        case .secretUnavailable(let detail):
            "Could not prepare the sandbox's secret: \(detail)"
        case .tooManyAddresses:
            "This Mac has more network addresses than the sandbox can block."
        case .hostAddressesUnknown:
            "TUFF could not read this Mac's network addresses, so it cannot tell the sandbox which to block."
        case .notHealthy(let detail):
            "The sandbox started but does not answer correctly: \(detail)"
        case .protectionNotVerified(let detail):
            "The sandbox's protection could not be verified, so it was not used: \(detail)"
        case .stoppedWhileStarting:
            "The sandbox was stopped while it was starting."
        case .notReady:
            "The sandbox is not ready."
        case .stillStarting:
            "The sandbox is still starting."
        case .noFoldersToMount:
            "There is no search folder for the file sandbox to read."
        case .unmountableFolder:
            "A search folder's path has a comma or a line break in it, which the container tool cannot mount."
        case .folderAccessDenied:
            "macOS did not let the file sandbox read a search folder. In System Settings, open Privacy & "
                + "Security, then Files and Folders (or, if it is not listed there, Full Disk Access), allow "
                + "\u{201C}container-runtime-linux\u{201D} for the folder, and try again."
        }
    }
}
#endif
