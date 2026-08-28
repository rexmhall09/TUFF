import TUFFAppCore
import SwiftUI

struct ModelLibraryColumnsView: View {
    let model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Models")
                    .font(.title3.weight(.semibold))
                Text("Grouped by how each one runs. Optional downloads live in "
                     + "the card of the model they belong to.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)

            ModelCatalogView(model: model, showsAddOns: true)
        }
    }
}
