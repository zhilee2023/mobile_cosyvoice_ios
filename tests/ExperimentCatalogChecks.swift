import Foundation

@main
struct ExperimentCatalogChecks {
    static func main() {
        let routes = ExperimentCatalog.routes
        precondition(Set(routes.map(\.id)).count == routes.count)
        let concluded: Set<String> = ["flow-buckets", "flow-thread-4", "flow-thread-2", "flow-thread-3",
            "llm-reuse", "llm-threads", "hift-threads", "flow-bf16", "adaptive-resident"]
        precondition(Set(ExperimentCatalog.archived.map(\.id)) == concluded)
        precondition(ExperimentCatalog.available.map(\.id) == ["llm-chain-reuse", "two-sentence-staged"])
        precondition(ExperimentCatalog.available.allSatisfy { $0.execution == .sentence })
        precondition(ExperimentCatalog.planned.isEmpty)
        precondition(ExperimentCatalog.defaultThreads == 4)
        precondition(routes.allSatisfy { !$0.conclusion.isEmpty })
        print("Experiment catalog PASS: concluded and unimplemented routes have no runnable buttons")
    }
}
