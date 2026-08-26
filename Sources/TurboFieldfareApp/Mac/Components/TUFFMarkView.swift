import SwiftUI
import TurboFieldfareMacPresentation

/// The TUFF mark: a bird built from one equilateral triangle and four quarter
/// circles of equal radius. Two quarters form the belly, one forms the
/// shoulder, one forms the raised wing, and the triangle is the beak.
struct TUFFMarkView: View {
    var body: some View {
        GeometryReader { proxy in
            let size = min(proxy.size.width, proxy.size.height)
            ZStack {
                RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                    .fill(.black)
                TUFFWingAndBeak().fill(TurboFieldfareMacTheme.markWingColor)
                TUFFBirdBody().fill(TurboFieldfareMacTheme.accentColor)
            }
            .frame(width: size, height: size)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .aspectRatio(1, contentMode: .fit)
        .accessibilityHidden(true)
    }
}

/// The raised wing quarter and the beak triangle.
private struct TUFFWingAndBeak: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: point(113.14, 128.19, in: rect))
        path.addCurve(
            to: point(420.19, 435.24, in: rect),
            control1: point(282.72, 128.19, in: rect),
            control2: point(420.19, 265.66, in: rect))
        path.addLine(to: point(113.14, 435.24, in: rect))
        path.closeSubpath()
        path.move(to: point(683.09, 281.72, in: rect))
        path.addLine(to: point(952, 435.24, in: rect))
        path.addLine(to: point(683.09, 588.76, in: rect))
        path.closeSubpath()
        return path
    }

    private func point(_ x: CGFloat, _ y: CGFloat, in rect: CGRect) -> CGPoint {
        CGPoint(
            x: rect.minX + (x / 1024) * rect.width,
            y: rect.minY + (y / 1024) * rect.height)
    }
}

/// The belly and shoulder quarter as one outline.
private struct TUFFBirdBody: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: point(72, 588.76, in: rect))
        path.addCurve(
            to: point(379.05, 895.81, in: rect),
            control1: point(72, 758.34, in: rect),
            control2: point(209.47, 895.81, in: rect))
        path.addCurve(
            to: point(686.1, 588.76, in: rect),
            control1: point(548.62, 895.81, in: rect),
            control2: point(686.1, 758.34, in: rect))
        path.addLine(to: point(686.09, 281.72, in: rect))
        path.addCurve(
            to: point(379.04, 588.76, in: rect),
            control1: point(516.51, 281.71, in: rect),
            control2: point(379.04, 419.18, in: rect))
        path.addLine(to: point(72, 588.76, in: rect))
        path.closeSubpath()
        return path
    }

    private func point(_ x: CGFloat, _ y: CGFloat, in rect: CGRect) -> CGPoint {
        CGPoint(
            x: rect.minX + (x / 1024) * rect.width,
            y: rect.minY + (y / 1024) * rect.height)
    }
}
