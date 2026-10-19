import Foundation

@main
struct PlaybackPlanChecks {
    static func main() {
        precondition(!PlaybackMode.buffered.mayStart(readyFiles: 0, producerFinished: true))
        precondition(!PlaybackMode.buffered.mayStart(readyFiles: 2, producerFinished: false))
        precondition(PlaybackMode.buffered.mayStart(readyFiles: 3, producerFinished: false))
        precondition(PlaybackMode.buffered.mayStart(readyFiles: 2, producerFinished: true))
        precondition(PlaybackMode.buffered.mayStart(readyFiles: 1, producerFinished: true))
        precondition(!PlaybackMode.complete.mayStart(readyFiles: 5, producerFinished: false))
        precondition(PlaybackMode.complete.mayStart(readyFiles: 5, producerFinished: true))
        precondition(PlaybackDiskBudget.mayAppend(files: 5, bytes: 1000, newBytes: 1000))
        precondition(!PlaybackDiskBudget.mayAppend(files: 6, bytes: 1000, newBytes: 1000))
        precondition(!PlaybackDiskBudget.mayAppend(files: 0, bytes: 0, newBytes: 0))
        precondition(!PlaybackDiskBudget.mayAppend(files: 1, bytes: PlaybackDiskBudget.maxBytes, newBytes: 1))
        precondition(PlaybackDiskBudget.mayAppend(files: 0, bytes: 0, newBytes: PlaybackDiskBudget.maxBytes))
        precondition(!PlaybackDiskBudget.mayAppend(files: 0, bytes: -1, newBytes: 1))
        // A constant 1.8 RTF producer cannot sustain first-sentence playback;
        // for five equal sentences, three completed sentences cover the deficit.
        let r = 1.8, sentence = 2.8
        let producerFinishAfterStart = 2 * r * sentence
        precondition(producerFinishAfterStart < 5 * sentence)
        precondition(r * sentence > sentence)
        // This is a scheduling model, NOT a device/audio-engine acceptance test.
        print("PASS: three-sentence preload, final short batch, complete mode, disk bounds, scheduling model")
    }
}
