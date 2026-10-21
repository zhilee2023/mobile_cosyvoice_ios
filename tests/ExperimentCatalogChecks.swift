import Foundation

@main
struct ExperimentCatalogChecks {
    static func main() {
        let routes = ExperimentCatalog.routes
        precondition(Set(routes.map(\.id)).count == routes.count)
        let concluded: Set<String> = ["flow-buckets", "flow-thread-4", "flow-thread-2", "flow-thread-3",
            "llm-reuse", "llm-threads", "hift-threads", "flow-bf16", "adaptive-resident",
            "llm-chain-reuse", "two-sentence-staged", "six-sentence-staged", "flow-metal-low-memory-256", "flow-metal-low-256", "flow-metal-low-384", "flow-metal-nan", "flow-metal-hybrid-256",
            "flow-w8-reference", "flow-w8-normal", "flow-w8-low", "flow-cpu-reference-384", "flow-metal-hybrid-384", "flow-cpu-reference-512", "flow-metal-hybrid-512",
            "three-sentence-staged-w8-head-full", "three-sentence-staged-w8-head-reduced", "flow-w8-profile", "flow-w8-metal-high", "three-sentence-staged", "three-sentence-staged-w8", "flow-projection-reference", "flow-projection-candidate", "three-sentence-staged-w8-head-reduced-reuse", "three-sentence-staged-projection-head-reuse"]
        precondition(Set(ExperimentCatalog.archived.map(\.id)) == concluded)
        precondition(ExperimentCatalog.available.filter { $0.execution == .sentence }.map(\.id) == ["three-sentence-staged-projection-head-reuse-hift-reuse", "three-sentence-staged-awq-head-reuse-hift-reuse"])
        precondition(ExperimentCatalog.available.filter { $0.execution == .module }.isEmpty)
        precondition(ExperimentCatalog.planned.isEmpty)
        precondition(ExperimentCatalog.defaultThreads == 4)
        precondition(routes.allSatisfy { !$0.conclusion.isEmpty })
        print("Experiment catalog PASS: concluded and unimplemented routes have no runnable buttons")
    }
}
