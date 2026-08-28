import TUFFAppCore
import SwiftUI

struct ModelLibraryColumnsView: View {
    let model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Models")
                    .font(.title3.weight(.semibold))
                Text("Each model includes its verified optional downloads in the same card.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)

            ModelCatalogView(model: model, showsAddOns: true)
        }
    }
}
