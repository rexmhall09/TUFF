import CoreGraphics

public enum AppWindowLayout {
    public static let minimumWidth: CGFloat = 820
    public static let minimumHeight: CGFloat = 560
    public static let defaultWidth: CGFloat = 1_080
    public static let defaultHeight: CGFloat = 720
    public static let sidebarMinimumWidth: CGFloat = 176
    public static let sidebarIdealWidth: CGFloat = 208
    public static let sidebarMaximumWidth: CGFloat = 240
    public static let detailMinimumWidth: CGFloat = 520

    public static var minimumHorizontalAllowance: CGFloat {
        sidebarMinimumWidth + detailMinimumWidth
    }
}
