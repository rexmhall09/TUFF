import CoreGraphics

public enum AppWindowLayout {
    public static let minimumWidth: CGFloat = 640
    public static let minimumHeight: CGFloat = 440
    public static let defaultWidth: CGFloat = 1_080
    public static let defaultHeight: CGFloat = 720
    public static let sidebarMinimumWidth: CGFloat = 160
    public static let sidebarIdealWidth: CGFloat = 196
    public static let sidebarMaximumWidth: CGFloat = 232
    public static let detailMinimumWidth: CGFloat = 420

    public static var minimumHorizontalAllowance: CGFloat {
        sidebarMinimumWidth + detailMinimumWidth
    }
}
