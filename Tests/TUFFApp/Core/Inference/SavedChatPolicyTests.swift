@testable import TUFFEngineAdapter
import Testing

/// When a chat's state is written to disk after a reply.
@Suite struct SavedChatPolicyTests {
    @Test func shortChatsAreNeverSaved() {
        #expect(!RealInferenceSession.shouldSaveChat(at: 1_023, saved: nil))
        #expect(RealInferenceSession.shouldSaveChat(at: 1_024, saved: nil))
    }

    /// A long chat is rewritten only after growing by a quarter, and at
    /// least 512 tokens, so it is saved a handful of times, not every reply.
    @Test func aSavedChatIsRewrittenOnlyAfterRealGrowth() {
        #expect(!RealInferenceSession.shouldSaveChat(at: 1_400, saved: 1_024))
        #expect(RealInferenceSession.shouldSaveChat(at: 1_536, saved: 1_024))
        #expect(!RealInferenceSession.shouldSaveChat(at: 4_900, saved: 4_000))
        #expect(RealInferenceSession.shouldSaveChat(at: 5_000, saved: 4_000))
        // The same state is never written twice.
        #expect(!RealInferenceSession.shouldSaveChat(at: 2_000, saved: 2_000))
    }

    @Test func savingCanBeTurnedOff() {
        #expect(RealInferenceSession.savesChats(environment: [:]))
        #expect(!RealInferenceSession.savesChats(environment: ["TUFF_SAVED_CHATS": "off"]))
    }
}
