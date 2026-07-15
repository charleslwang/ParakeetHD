import Foundation

public struct ParakeetTokenizer {
    public let tokens: [String]

    public init(tokensURL: URL) throws {
        let text = try String(contentsOf: tokensURL, encoding: .utf8)
        self.tokens = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    }

    public func decode(tokenIDs: [Int]) -> String {
        let pieces = tokenIDs.compactMap { id -> String? in
            guard id >= 0 && id < tokens.count else { return nil }
            return tokens[id]
        }
        let joined = pieces.joined().replacingOccurrences(of: "▁", with: " ")
        return joined
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
