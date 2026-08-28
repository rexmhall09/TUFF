import TUFFAppCore
import SwiftUI

struct ModelLibraryColumnsView: View {
    let model: AppModel

    var body: some View {
        // No heading of its own: the workspace already has one that says
        // "Models", and a second one under it was the same word twice.
        ModelCatalogView(model: model, showsAddOns: true)
    }
}
