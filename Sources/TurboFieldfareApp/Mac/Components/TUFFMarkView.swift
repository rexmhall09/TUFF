import SwiftUI
import TurboFieldfareMacPresentation

struct TUFFMarkView: View {
    var body: some View {
        GeometryReader { proxy in
            let size = min(proxy.size.width, proxy.size.height)
            ZStack {
                TUFFFacet(points: [
                    CGPoint(x: 0.16, y: 0.20),
                    CGPoint(x: 0.50, y: 0.20),
                    CGPoint(x: 0.33, y: 0.50),
                ])
                .fill(.primary)

                TUFFFacet(points: [
                    CGPoint(x: 0.50, y: 0.20),
                    CGPoint(x: 0.84, y: 0.20),
                    CGPoint(x: 0.67, y: 0.50),
                ])
                .fill(TurboFieldfareMacTheme.accentColor)

                TUFFFacet(points: [
                    CGPoint(x: 0.33, y: 0.50),
                    CGPoint(x: 0.67, y: 0.50),
                    CGPoint(x: 0.50, y: 0.80),
                ])
                .fill(.primary)
            }
            .frame(width: size, height: size)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .aspectRatio(1, contentMode: .fit)
        .accessibilityHidden(true)
    }
}

private struct TUFFFacet: Shape {
    let points: [CGPoint]

    func path(in rect: CGRect) -> Path {
        guard let first = points.first else { return Path() }
        var path = Path()
        path.move(to: point(first, in: rect))
        for value in points.dropFirst() {
            path.addLine(to: point(value, in: rect))
        }
        path.closeSubpath()
        return path
    }

    private func point(_ value: CGPoint, in rect: CGRect) -> CGPoint {
        CGPoint(
            x: rect.minX + value.x * rect.width,
            y: rect.minY + value.y * rect.height)
    }
}
